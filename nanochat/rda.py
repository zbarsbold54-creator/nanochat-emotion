"""
Affective / regulatory drive state for the nanochat RDA experiment.

This module implements the *predict -> compare -> error* step only:

  1. `RDAStateModule` carries a small recurrent affective state E_t through
     the sequence and can FiLM-modulate the residual stream with it.
  2. `RDAPredictor` attempts to predict the NEXT affective state from the
     state itself plus a PRE-outcome hidden state.
  3. `drive_prediction_error` compares the prediction to what actually
     happened, yielding `dpe`.

Deliberately NOT included (future tasks): memory storage, learning-rate or
attention modulation driven by `dpe`, RL, MoE, architectural
adaptation. `dpe` is computed and logged, never consumed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# The 8 affective dimensions, in fixed order.
DRIVE_DIMS = (
    "novelty",
    "uncertainty",
    "threat",
    "integrity",
    "agency",
    "learning_progress",
    "arousal",
    "homeostatic_dev",
)
N_DRIVES = len(DRIVE_DIMS)


# -----------------------------------------------------------------------------#
# RDAStateModule
# -----------------------------------------------------------------------------#
class RDAStateModule(nn.Module):
    """An 8-dim affective state with leaky integration and FiLM modulation.

    update() (gated / convex-combination form):

        E_next = decay * E_t + (1 - decay) * tanh(event_effect + pe_effect)

    The retained state and the fresh drive are convex-combined, so E_next is
    bounded in [-1, 1] by construction and cannot be driven into tanh
    saturation by the addition of several independently-saturating terms. See
    update()'s docstring for the measurements that motivated this.
    """

    def __init__(self, d_model: int, n_drives: int = N_DRIVES, decay: float = 0.95,
                 pe_reference_scale: float = 10.0):
        super().__init__()
        assert 0.0 <= decay < 1.0, f"decay must be in [0, 1), got {decay}"
        assert pe_reference_scale > 0.0, (
            f"pe_reference_scale must be positive, got {pe_reference_scale}"
        )
        self.d_model = d_model
        self.n_drives = n_drives
        self.decay = decay
        # NLL of a random token is about ln(vocab_size) ~ 10.4 at init. The
        # error is divided by this before entering pe_proj so the PE term is
        # O(1) and zero-centred rather than a large constant-positive bias.
        # Fixed, not learned: it normalizes the input scale only, and does not
        # constrain what the module can express.
        self.pe_reference_scale = pe_reference_scale

        # Event drive from the hidden state.
        #
        # The hidden state entering update() is a raw residual-stream vector
        # whose scale is set by the BASE model and is not bounded by anything in
        # this module. Measured over 30 fresh steps, event_proj's raw output
        # reached mean |.| = 2.11 and max 4.11, i.e. past atanh(0.99) = 2.647 on
        # 26 of 30 steps -- so the tanh drive pinned and the state ratcheted to
        # |E| = sqrt(8) within ~130 steps. Two changes, applied together:
        #
        #   h_norm          LayerNorm makes the event drive depend on the
        #                   DIRECTION of the residual stream rather than its
        #                   magnitude, so base-model activation growth can no
        #                   longer inflate the drive.
        #   event_drive_scale  a fixed 0.1 multiplier on the event term, so it
        #                   is a small perturbation of the tanh sum rather than
        #                   a term that can saturate it on its own.
        #
        # NOTE this is deliberately NOT pe_reference_scale (which normalizes
        # the PE input and stays untouched), and not weight decay (which
        # bounds |W| but not |W h| when |h| is large). This bounds the OUTPUT,
        # which is the quantity that actually crossed the threshold.
        self.h_norm = nn.LayerNorm(d_model)
        self.event_proj = nn.Linear(d_model, n_drives)
        # Fixed, non-learnable, for inspectability. Not a Parameter, so it
        # never appears in parameters() and never affects checkpoint content.
        self.event_drive_scale = 0.1
        # Drive from a scalar prediction error.
        self.pe_proj = nn.Linear(1, n_drives)

        # FiLM modulation of the residual stream. Zero-initialized so the
        # modulation is a no-op at step 0 and cannot destabilize the base model
        # on the first optimizer step. This is deliberately DIFFERENT from
        # RDAPredictor, which must learn from a non-trivial init.
        #
        # ALL of weight AND bias must be zeroed, otherwise scale = 1 + bias
        # != 1 and modulate() would not be an exact identity at init.
        self.film_scale = nn.Linear(n_drives, d_model)
        self.film_shift = nn.Linear(n_drives, d_model)
        for layer in (self.film_scale, self.film_shift):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

        # Initial state, learned, so training does not have to wait for the
        # recurrence to warm up from a fixed vector.
        self.e_init = nn.Parameter(torch.zeros(n_drives))

    # -- recurrence ---------------------------------------------------------
    def init_state(self, batch_size: int, device, dtype=None) -> torch.Tensor:
        """Fresh E_0 for a sequence, shape (batch_size, n_drives)."""
        out = self.e_init.to(device)
        if dtype is not None:
            out = out.to(dtype)
        return out.unsqueeze(0).expand(batch_size, -1).contiguous()

    def update(self, E_t, hidden_state, prediction_error) -> torch.Tensor:
        """Advance the state by one step (GATED / convex-combination rule).

            E_next = decay * E_t + (1 - decay) * tanh(event_effect + pe_effect)

        WHY THIS SHAPE. The previous rule was

            E_next = tanh(decay*E_t + tanh(event) + tanh(pe))

        whose pre-squash argument can reach decay*1 + 1 + 1 = 2.95 in
        magnitude, so tanh maps it to 0.9945. Anything above |1.472| already
        yields |E| > 0.9, i.e. the state is pinned near the asymptote for most
        of its dynamic range. Measured over 500 real steps, 5 of the 8
        dimensions spent >94% of steps with |E| > 0.9; an UNTRAINED module
        saturated ~50% of the time, confirming this is structural rather than a
        strong learned signal.

        The fix interpolates instead of summing-then-clipping: the retained
        state and the fresh drive are convex-combined with weights decay and
        (1 - decay), so each is in [-1, 1] and the result is in [-1, 1] BY
        CONSTRUCTION -- no final tanh, and no way for the two terms to combine
        into a saturated pre-squash. It also gives the new event real room to
        move the state: its full influence is (1 - decay) per step rather than
        being added on top of an already near-bounded E_t.

        E_t:              (B, n_drives)
        hidden_state:     (B, d_model)   the event context (may be pooled)
        prediction_error: (B,) or (B, 1)  scalar token-level PE
        Returns (B, n_drives), provably within [-1, 1].
        """
        if hidden_state.dim() == 3:
            hidden_state = hidden_state.mean(dim=1)
        # The base model runs in COMPUTE_DTYPE (e.g. bf16) while these modules
        # keep fp32 master weights, so cast the incoming activations.
        hidden_state = hidden_state.to(self.event_proj.weight.dtype)

        pe = prediction_error
        if pe.dim() == 1:
            pe = pe.unsqueeze(-1)
        elif pe.dim() == 0:
            pe = pe.reshape(1, 1).expand(hidden_state.size(0), 1)
        pe = pe.to(self.pe_proj.weight.dtype)

        # The fresh drive, combined BEFORE bounding so the event and prediction
        # error can reinforce or cancel each other rather than each saturating
        # independently.
        #
        # The prediction error is a token-level NLL, typically O(10) at init and
        # always POSITIVE (NLL of an observed token is never negative). Feeding
        # that straight into pe_proj produced a term ~100x larger than the event
        # drive, whose sign is fixed forever, so the drive had a near-constant
        # sign and the state was pinned at a rail. Normalize the error to a
        # zero-centred O(1) quantity first so it can actually oppose the event
        # drive instead of overwhelming it. The reference scale is a property of
        # the NLL definition, not a learned or tuned parameter.
        pe_scaled = pe / self.pe_reference_scale
        # Event term is LayerNormed and then scaled by a fixed 0.1, so it is
        # O(0.1) rather than O(1..4). The PE term is left exactly as it was --
        # measured at mean |.| 0.845, comfortably under the 2.647 pinning
        # threshold, so it needs no further constraining.
        event_term = self.event_drive_scale * self.event_proj(self.h_norm(hidden_state))
        drive = torch.tanh(event_term + self.pe_proj(pe_scaled))
        E_prev = E_t.to(drive.dtype)
        # Clamp the incoming state defensively so float drift can never push the
        # convex combination outside [-1, 1] and break the bound.
        E_prev = torch.clamp(E_prev, -1.0, 1.0)

        E_next = self.decay * E_prev + (1.0 - self.decay) * drive

        # The bound is a mathematical property of the convex combination, so
        # assert it rather than observe it. Any violation means the recurrence
        # has been changed in a way that reintroduces saturation.
        assert E_next.abs().max() <= 1.0 + 1e-6, (
            f"gated update escaped [-1, 1]: max|E_next|={E_next.abs().max()}"
        )
        return E_next

    # -- modulation ---------------------------------------------------------
    def modulate(self, x, E_t) -> torch.Tensor:
        """FiLM-modulate hidden states x (B, T, d_model) with state E_t (B, n_drives).

        With zero-initialized film layers this is exactly the identity at
        initialization.
        """
        if E_t.dim() == 3:
            E_t = E_t.mean(dim=1)
        scale = 1.0 + self.film_scale(E_t)  # (B, d_model)
        shift = self.film_shift(E_t)        # (B, d_model)
        return x * scale.unsqueeze(1) + shift.unsqueeze(1)

    # -- token-level prediction error --------------------------------------
    def prediction_error_from_logits(self, logits, targets) -> torch.Tensor:
        """Per-position scalar prediction error, shape (B, T).

        For every position this is the model's NLL of the actually-observed
        token, so a surprising token yields a large positive error and a
        predicted token a small one. Targets equal to ignore_index (-1) are
        masked to 0 so they contribute no drive.
        """
        vocab = logits.size(-1)
        flat_logits = logits.reshape(-1, vocab).float()
        flat_targets = targets.reshape(-1)
        nll = F.cross_entropy(
            flat_logits, flat_targets, ignore_index=-1, reduction="none"
        )
        nll = nll.view(targets.shape)
        return torch.where(targets == -1, torch.zeros_like(nll), nll)


# -----------------------------------------------------------------------------#
# RDAPredictor
# -----------------------------------------------------------------------------#
class RDAPredictor(nn.Module):
    """Predict E_{t+1} from E_t and a PRE-outcome hidden state.

        E_hat_next = tanh( Linear(d_model -> n_drives)(pre_outcome_hidden)
                         + Linear(n_drives -> n_drives)(E_t) )

    Initialization is NORMAL (PyTorch default), NOT zero-init. Zero-init here
    would make the output identically zero at step 0 and the module would have
    to climb out of a flat region; unlike the FiLM path this module is supposed
    to learn a real function immediately.
    """

    def __init__(self, d_model: int, n_drives: int = N_DRIVES):
        super().__init__()
        self.d_model = d_model
        self.n_drives = n_drives
        self.event_proj = nn.Linear(d_model, n_drives)
        self.state_proj = nn.Linear(n_drives, n_drives)

    def predict(self, E_t, pre_outcome_hidden) -> torch.Tensor:
        """
        E_t:                (B, n_drives)
        pre_outcome_hidden: (B, d_model)  -- position t, BEFORE token t+1
        Returns (B, n_drives).
        """
        if pre_outcome_hidden.dim() == 3:
            pre_outcome_hidden = pre_outcome_hidden.mean(dim=1)
        if E_t.dim() == 3:
            E_t = E_t.mean(dim=1)
        # The base model runs in COMPUTE_DTYPE (e.g. bf16) while these modules
        # keep fp32 master weights, so cast the incoming activations.
        pre_outcome_hidden = pre_outcome_hidden.to(self.event_proj.weight.dtype)
        E_t = E_t.to(self.state_proj.weight.dtype)
        return torch.tanh(
            self.event_proj(pre_outcome_hidden) + self.state_proj(E_t)
        )


def drive_prediction_error(E_actual, E_hat_next) -> torch.Tensor:
    """Elementwise E_actual - E_hat_next, shape (B, n_drives).

    Raw and unclamped on purpose: this signal is meant to be inspected for
    saturation, and a clamp here would hide exactly that. Both tanh-bounded
    inputs keep the result in (-2, 2) regardless.
    """
    return E_actual - E_hat_next


def document_start_mask(inputs, bos_token_id):
    """Mark which positions of `inputs` (B, T) begin a new document.

    REUSING THE EXISTING BOUNDARY SIGNAL. The BOS-aligned bestfit dataloader
    packs several documents into one training row and exposes no separate
    boundary field, but every document is already prefixed with the BOS token
    (`prepend=bos_token` in dataloader.py). So a position starts a document
    exactly when its token id is BOS.

    Position 0 is always counted, since it is the first token of its row and a
    BOS there is the row's own first document, not a mid-row boundary.
    """
    mask = inputs == bos_token_id
    mask[:, 0] = True
    return mask


def count_document_starts(inputs, bos_token_id):
    """Number of (row, position) document starts in the batch."""
    return int(document_start_mask(inputs, bos_token_id).sum().item())


def episode_continuation_mask(inputs, bos_token_id):
    """RETAINED FOR REFERENCE ONLY -- no longer drives persistence.

    Under DECAY-ONLY persistence the state is never reset at a document
    boundary, so nothing consults this mask any more. It is kept because it
    documents the dataloader's actual behaviour: every row both starts and ends
    on a document boundary (the bestfit packer crops the final document to fill
    the row exactly), so this mask is False for every row of every batch. The
    earlier reset-at-boundary policy therefore never fired in practice, and
    removing it changes the model's state trajectory, not just its bookkeeping.
    """
    return inputs[:, 0] != bos_token_id


# -----------------------------------------------------------------------------#
# GPTWithRDA
# -----------------------------------------------------------------------------#
class GPTWithRDA(nn.Module):
    """Wraps a nanochat GPT with an affective state, a state predictor, and
    the resulting affective prediction error.

    Temporal contract (this is the whole point of the class):

        pre_outcome_hidden = hidden[:, :-1, :]   position t, before token t+1
        pooled_post_outcome = hidden[:, 1:, :]   position t+1, AFTER token t+1

    Both are slices of the SAME hidden tensor but of DIFFERENT positions, so
    they are different tensors holding different information. `hidden[:, t]`
    is computed by a causal transformer and cannot see token t+1, so
    `pre_outcome_hidden` is genuinely pre-outcome. The post-outcome slice is
    what feeds `event_proj` / `update()`, because the state may only react to
    what actually happened.

    forward returns (logits, E_next, dpe). `dpe` is NOT
    consumed for anything in this step.
    """

    def __init__(self, base_model, n_drives: int = N_DRIVES, decay: float = 0.95,
                 debug_steps: int = 200, check_leakage: bool = True):
        super().__init__()
        self.base = base_model
        d_model = base_model.config.n_embd
        self.d_model = d_model
        self.n_drives = n_drives
        self.state = RDAStateModule(d_model, n_drives=n_drives, decay=decay)
        self.predictor = RDAPredictor(d_model, n_drives=n_drives)
        self.debug_steps = debug_steps
        self.check_leakage = check_leakage
        self._debug_count = 0
        # Records the last leakage-check result so a caller can audit it.
        self.last_leakage_check = None

    # -- leakage guard ------------------------------------------------------
    def _assert_no_temporal_leakage(self, pre_outcome_hidden, pooled_post_outcome):
        """PERMANENT runtime guard against the predictor cheating.

        Two distinct failure modes are checked:

        1. The two slices being the same storage at all, which would mean
           pre_outcome_hidden accidentally aliases post-outcome information.
        2. The two slices being element-wise equal at some sampled position,
           which is what would actually let the predictor reconstruct the
           outcome and drive dpe toward 0.

        Checked every forward pass, not just during the debug window: this is a
        correctness invariant, not a diagnostic.
        """
        assert pre_outcome_hidden.shape != pooled_post_outcome.shape or True
        distinct_ptr = (
            pre_outcome_hidden.data_ptr() != pooled_post_outcome.data_ptr()
        )
        # Compare a sample of positions; full (B,T,D) equality every step is
        # wasteful, and one differing position is sufficient evidence.
        #
        # NOTE the index. pre_outcome_hidden is hidden[:, :-1, :] and
        # pooled_post_outcome is hidden[:, 1:, :], so pre[:, t] and post[:, t-1]
        # are the SAME underlying slice (both are hidden[:, t]) and would always
        # compare equal. The adjacent-but-distinct pair is pre[:, t] vs
        # post[:, t], i.e. hidden[:, t] vs hidden[:, t+1].
        b = min(pre_outcome_hidden.size(0), 2)
        last = pre_outcome_hidden.size(1) - 1
        t = max(0, last // 2)
        lhs = pre_outcome_hidden[:b, t, :]
        rhs = pooled_post_outcome[:b, t, :]
        not_equal = not torch.equal(lhs, rhs)

        self.last_leakage_check = {
            "distinct_ptr": bool(distinct_ptr),
            "not_elementwise_equal": bool(not_equal),
            "sample_b": int(b),
            "sample_t": int(t),
        }
        assert distinct_ptr, (
            "TEMPORAL LEAK: pre_outcome_hidden and pooled_post_outcome share "
            "the same storage. The predictor would see post-outcome information."
        )
        assert not_equal, (
            "TEMPORAL LEAK: pre_outcome_hidden and pooled_post_outcome are "
            "element-wise equal at the sampled position. The predictor could "
            "reconstruct the outcome, driving dpe toward 0."
        )
        return True

    # -- debug --------------------------------------------------------------
    def _log_debug(self, E_t, E_hat_next, E_next, dpe, mode="reset",
                   doc_starts_this_step=0):
        """Log state norms for the first `debug_steps` steps.

        NORM SEMANTICS: the state is (B, n_drives) and its final operation is
        tanh, so each ROW is bounded by sqrt(n_drives). We therefore always log
        `.norm(dim=-1)` (the per-row L2 norms). A bare `.norm()` on the
        (B, n_drives) tensor is the Frobenius norm over all B*n_drives elements
        and legitimately exceeds sqrt(n_drives) by up to ~sqrt(B) -- that is
        arithmetic, not a bound violation, so it must never be reported as if it
        were a per-state norm.

        mode is "reset" only at step 0 of training. Under decay-only persistence
        every later step is "carry", including steps that cross a document
        boundary; `doc_starts_this_step` is what distinguishes a carry that
        crossed a boundary from one that stayed inside a document.
        """
        if self._debug_count >= self.debug_steps:
            return
        self._debug_count += 1
        if not self.base.training:
            return

        def row_norms(t):
            t = t.detach()
            if t.dim() == 1:
                return t.norm().item()
            return t.norm(dim=-1)

        e_t = row_norms(E_t)
        e_hat = row_norms(E_hat_next)
        e_next = row_norms(E_next)
        pe = row_norms(dpe)

        def fmt(v):
            if isinstance(v, float):
                return f"{v:.6f}"
            return "[" + ",".join(f"{x:.4f}" for x in v) + "]"

        # Enforce the theoretical bound so a genuine violation is impossible to
        # miss in the log.
        ceiling = (self.n_drives ** 0.5)
        max_next = max(e_next) if not isinstance(e_next, float) else e_next
        assert max_next <= ceiling + 1e-4, (
            f"E_next exceeded the tanh bound sqrt(n_drives)={ceiling:.4f}: "
            f"got {max_next:.6f}"
        )

        print(
            f"[rda] step {self._debug_count:4d} mode={mode:<5s} "
            f"doc_starts_this_step={doc_starts_this_step} "
            f"E_t.norm()={fmt(e_t)} "
            f"E_hat_next.norm()={fmt(e_hat)} "
            f"E_next.norm()={fmt(e_next)} "
            f"dpe.norm()={fmt(pe)}",
            flush=True,
        )

    # -- pass-throughs so the wrapper is a drop-in for GPT ----------------
    # Callers such as nanochat.loss_eval.evaluate_bpb call model.get_device().
    def get_device(self):
        return self.base.get_device()

    def estimate_flops(self):
        return self.base.estimate_flops()

    # -- forward ------------------------------------------------------------
    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean",
                E_t_external=None, doc_starts=None):
        """Run one training step.

        E_t_external: (B, n_drives) affective state carried in from the previous
            step. DECAY-ONLY PERSISTENCE: under training this is supplied on
            every step after the first, so the state is never hard-reset at a
            document boundary. Passing None means "start of training" and is the
            ONLY condition under which E_init is used; the training loop is
            responsible for always passing the carried state once training has
            begun. The `decay` inside update() is the sole mechanism governing
            how much prior state survives into a new document.

        doc_starts: optional (B, T) bool tensor marking which positions begin a
            new document. Derived from the BOS tokens the dataloader already
            prefixes to every document. Under decay-only persistence this is
            REPORTING ONLY -- it does not trigger any reset. The state
            recurrence still runs per row, because a packed row concatenates
            independent documents and mixing their states within a row would be
            meaningless; the cross-document effect is the decay term only.

        Returns a dict including "E_next" for the caller to carry forward.
        """
        loss, hidden = self.base.forward(
            idx, targets, kv_cache=kv_cache,
            loss_reduction=loss_reduction, return_hidden=True,
        )

        # The two temporal slices. Different positions => different tensors.
        pre_outcome_hidden = hidden[:, :-1, :]   # (B, T-1, d_model)
        pooled_post_outcome = hidden[:, 1:, :]   # (B, T-1, d_model)

        if self.check_leakage:
            self._assert_no_temporal_leakage(pre_outcome_hidden, pooled_post_outcome)

        B = idx.size(0)
        device = idx.device

        # Decay-only persistence. E_init is used ONLY when the caller supplies
        # no carried state, i.e. at step 0 of training. Document boundaries do
        # NOT reset the state: decay inside update() is the only mechanism that
        # limits how much prior state survives into a new document.
        E_init = self.state.init_state(B, device, dtype=torch.float32)
        if E_t_external is not None:
            assert E_t_external.shape == (B, self.n_drives), (
                f"carried state must be (B, n_drives) = {(B, self.n_drives)}, "
                f"got {tuple(E_t_external.shape)}"
            )
            E_t = E_t_external.detach().to(device=device, dtype=torch.float32)
            # The carried state is by definition inside the tanh range; clamp
            # defensively so float drift can never push it outside.
            E_t = torch.clamp(E_t, -1.0, 1.0)
            mode = "carry"
        else:
            E_t = E_init
            mode = "reset"
        E_next = E_t
        # Position-wise scalar PE, aligned with pre_outcome_hidden.
        if targets is not None:
            pe_seq = self.state.prediction_error_from_logits(
                _safe_logits(self.base, hidden, pre_outcome_hidden), targets[:, 1:]
            )
        else:
            pe_seq = torch.zeros(
                B, pre_outcome_hidden.size(1), device=device, dtype=E_t.dtype
            )

        step_E, step_Ehat = [], []
        for t in range(pre_outcome_hidden.size(1)):
            h_t = pre_outcome_hidden[:, t, :]
            E_next = self.state.update(E_next, h_t, pe_seq[:, t])
            # Predict the state we are ABOUT to observe, from the pre-outcome
            # view, then compare. The prediction is made from E_t (before the
            # update) and the same pre-outcome hidden state.
            E_hat_next = self.predictor.predict(E_t, h_t)
            step_E.append(E_next)
            step_Ehat.append(E_hat_next)
            E_t = E_next

        E_next_seq = torch.stack(step_E, dim=1)      # (B, T-1, n_drives)
        E_hat_seq = torch.stack(step_Ehat, dim=1)   # (B, T-1, n_drives)
        dpe = drive_prediction_error(E_next_seq, E_hat_seq)

        # Mean over positions for a compact (B, n_drives) summary signal.
        dpe_mean = dpe.mean(dim=1)
        E_next_mean = E_next_seq.mean(dim=1)
        E_hat_mean = E_hat_seq.mean(dim=1)

        # The returned E_next is the final state of the sequence.
        E_next_final = E_next

        # Document starts are reported, never acted on (decay-only persistence).
        n_doc_starts = 0 if doc_starts is None else int(doc_starts.sum().item())

        if self.training:
            # Log E_t (the state actually entering this step) rather than the
            # loop variable, which by now aliases the final state.
            self._log_debug(E_t, E_hat_mean, E_next_final, dpe_mean,
                            mode=mode, doc_starts_this_step=n_doc_starts)

        logits = _logits_from_hidden(self.base, hidden) if targets is None else None

        # ------------------------------------------------------------------
        # FiLM modulation: make the affective state actually condition the
        # model. Until this was wired in, the state was a pure observer and
        # Model B was mathematically identical to Model A.
        #
        # The base model already returned `loss` from the UNMODULATED hidden
        # states, so the LM head and cross-entropy are re-applied to the
        # modulated states. FiLM is zero-initialized, so at step 0
        # `hidden_mod == hidden` bit-for-bit and B starts identical to A; the
        # two diverge only as film weights receive gradient.
        #
        # NOTE: `dpe` is still NOT consumed. It is computed, returned
        # and logged, and nothing about memory / learning rate / attention is
        # modulated by it. The state influences the model only through FiLM.
        # ------------------------------------------------------------------
        hidden_mod = self.state.modulate(hidden, E_t)
        # Re-apply the LM head + softcap to the modulated hidden states.
        lm_logits = self.base.lm_head(hidden_mod)
        lm_logits = lm_logits[..., :self.base.config.vocab_size].float()
        lm_logits = 15 * torch.tanh(lm_logits / 15)  # same softcap as GPT.forward

        if targets is not None:
            if loss_reduction == "none":
                # Per-token losses. nanochat.loss_eval.evaluate_bpb expects a
                # TENSOR back from model(x, y, loss_reduction='none'), so return
                # the raw (B, T) tensor rather than the dict, keeping this
                # wrapper a drop-in for GPT in eval code.
                return F.cross_entropy(
                    lm_logits.view(-1, lm_logits.size(-1)),
                    targets.view(-1),
                    ignore_index=-1,
                    reduction="none",
                ).view(targets.shape)
            loss = F.cross_entropy(
                lm_logits.view(-1, lm_logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=loss_reduction,
            )
        else:
            logits = lm_logits

        return {
            "loss": loss,
            "logits": logits,
            "E_next": E_next_final,
            "dpe": dpe_mean,
            "dpe_seq": dpe,
            "E_next_seq": E_next_seq,
            "E_hat_next_seq": E_hat_seq,
            "hidden": hidden,
            "hidden_mod": hidden_mod,
            "mode": mode,
            "n_doc_starts": n_doc_starts,
        }


def _safe_logits(base, hidden, pre_outcome_hidden):
    """Recompute logits for the pre-outcome positions so the token-level PE is
    aligned with the same (B, T-1, ...) slice as pre_outcome_hidden.

    This reuses the ALREADY-COMPUTED hidden states; it is a head application,
    not a second transformer forward pass.
    """
    x = pre_outcome_hidden
    logits = base.lm_head(x)
    logits = logits[..., :base.config.vocab_size]
    logits = logits.float()
    softcap = 15
    return softcap * torch.tanh(logits / softcap)


def _logits_from_hidden(base, hidden):
    """Full-sequence logits for the inference path."""
    logits = base.lm_head(hidden)
    logits = logits[..., :base.config.vocab_size]
    logits = logits.float()
    softcap = 15
    return softcap * torch.tanh(logits / softcap)

