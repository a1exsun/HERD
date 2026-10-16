import { Eq, MathDetails } from '../components/MathDetails.tsx'
import { Tex } from '../components/Tex.tsx'
import { LoraMerge } from '../demos/LoraMerge.tsx'
import { PolicyViews } from '../demos/PolicyViews.tsx'
import { tx } from '../lib/tex.ts'

export function DriftConstrainedDistillation() {
  return (
    <div className="subsection" id="drift-constrained-distillation">
      <h3 className="reveal">Drift-Constrained Distillation</h3>
      <div className="text reveal">
        <p>
          HERD updates the deployed predictor exclusively through drift-constrained distillation. The method distills
          the behavior that the latest harness exposes while holding the student within two behavioral drift budgets
          around complementary anchors. Parameter inheritance is recursive across the full stream:{' '}
          <Tex tex={tx`\bar\theta_{t-1}`} /> initializes stage <Tex tex="t" />, and{' '}
          <Tex tex={tx`\bar\theta_0=\theta_0`} />.
        </p>
      </div>

      <div className="para">
        <h4 className="reveal">Teacher and student</h4>
        <div className="text reveal">
          <p>
            For a training record <Tex tex={tx`(x,y^\star)`} />, the teacher <Tex tex="p^T" /> is the frozen previous
            checkpoint under privileged context: the routed skill as the system message, the prompt{' '}
            <Tex tex="x" />, the verified answer <Tex tex={tx`y^\star`} /> replayed as the teacher&rsquo;s own previous
            turn, and the request &ldquo;Repeat your previous answer exactly, with no other text.&rdquo; The student{' '}
            <Tex tex="p^S" /> is the same checkpoint with a fresh LoRA and sees only the bare prompt. The two views
            differ only in the harness and the answer, so the student learns what the harness adds.
          </p>
          <p>
            The replay conditions the teacher strongly without any wording the student could copy. An answer announced
            in the prompt (&ldquo;The verified answer is &hellip;&rdquo;) invites the student to learn the
            announcement, and an answer placed in the system message is largely ignored. The loss is the
            full-vocabulary forward KL divergence from teacher to student at every position of the verified answer,
            averaged per token within each sequence and then over sequences. Taking it on the verified answer&rsquo;s
            own prefix rather than on student samples closes degenerate paths, such as prompt echo, on which teacher
            and student agree and the gradient vanishes; no rollouts are generated. Historical residuals are never
            replayed as training targets, and no test example enters evolution, distillation, or drift estimation.
          </p>
        </div>
        <PolicyViews />
      </div>

      <div className="para">
        <h4 className="reveal">Drift constraints</h4>
        <div className="text reveal">
          <p>
            Checkpoint drift is the KL divergence from the frozen previous checkpoint without harness,{' '}
            <Tex tex="p^C" />, to the student, measured on <Tex tex={tx`\bar\theta_{t-1}`} />
            &rsquo;s greedy answers to up to eight probes per task retained in{' '}
            <Tex tex={tx`\mathcal R_{t-1}^{\mathrm{task}}`} />, grouped by task. Base drift is the KL divergence from the
            immutable base model <Tex tex="p^0" /> to the student, measured on <Tex tex={tx`\theta_0`} />
            &rsquo;s greedy answers to <Tex tex={tx`\mathcal G`} />, grouped by source. The checkpoint anchor limits
            each transition, while the base anchor provides a fixed reference against drift that accumulates over many
            transitions.
          </p>
          <p>
            Each probe completion is the anchor&rsquo;s first 32 greedy tokens: the opening of a response decides its
            mode (reasoning or answering directly, format, preamble), and a longer average would hide a flip of that
            first decision. Each drift is the <em>worst group&rsquo;s</em> mean KL, because a mean over all probes
            would let one task or one general source drift far while the overall mean stays within budget.{' '}
            <Tex tex={tx`\mathcal G`} /> holds open-ended reasoning and instruction following, where drift is
            capability loss; multiple-choice records are left out, since anchoring the base model&rsquo;s verbose
            replies there would pin answer format rather than capability.
          </p>
        </div>
      </div>

      <div className="para">
        <h4 className="reveal">Lagrangian optimization</h4>
        <div className="text reveal">
          <p>
            HERD writes the drift limits as budgets rather than penalties, so one budget keeps the same meaning at
            every stage of the stream. Each budget <Tex tex={tx`\epsilon_C=\epsilon_G=0.05`} /> nats per token is held
            by a Lagrange multiplier updated by bounded dual ascent. While a multiplier is positive, the worst
            group&rsquo;s KL enters the update with weight <Tex tex={tx`\lambda_j/\epsilon_j`} />; a group above budget
            raises its multiplier, and slack lowers it towards zero, where the anchor has no effect.
          </p>
          <p>
            Each stage runs 32 updates of 32 records with AdamW at learning rate <Tex tex="10^{-4}" />, decaying linearly
            to zero, and gradient clipping at 1.0. The final iterate is kept; the decay makes it reflect the whole stage
            rather than the label mix of its last minibatches. The anchors&rsquo; hidden states on the fixed probe
            sequences are computed once, so no second model is resident during training. The fixed update
            count and probe sizes make per-stage cost independent of <Tex tex="t" />, giving <Tex tex="O(T)" /> total
            update cost.
          </p>
        </div>
      </div>

      <div className="para">
        <h4 className="reveal">Parameterization</h4>
        <div className="text reveal">
          <p>
            Within a stage, the inherited dense weights remain frozen and only a rank-32 LoRA (alpha 64, no dropout) on
            every linear layer of the text backbone is trained. The distillation loss and the multiplier-weighted drift
            terms backpropagate into the current factors only, and teacher and anchor targets stop gradients. At the
            end of the stage, the bf16 low-rank update is merged into the inherited dense weights, and drift is
            measured again and recorded with the stage. The next stage freezes this merged checkpoint and initializes
            fresh factors with a zero effective update. The merged predictor deploys without the harness. The drift
            constraints compare output distributions, not weight-space distances.
          </p>
        </div>
        <LoraMerge />
      </div>

      <MathDetails title="Equations for Drift-Constrained Distillation" count={7}>
        <Eq
          name="Teacher and student"
          tex={tx`\begin{aligned}p_{t,k}^{T}&=\pi_{\bar\theta_{t-1}}\bigl(\cdot\mid \Omega_{\mathcal F}(K_t,x),\,x,\,y^\star,\,\rho,\,y^\star_{<k}\bigr),\\ p_{t,k}^{S}&=\pi_\vartheta\bigl(\cdot\mid x,\,y^\star_{<k}\bigr)\end{aligned}`}
        >
          <Tex tex={tx`\rho`} /> is the repeat request. The teacher sees <Tex tex={tx`y^\star`} /> as its own previous
          turn; both views are evaluated on the prefixes of the verified answer.
        </Eq>
        <Eq
          name="Distillation objective"
          tex={tx`\mathcal L_t^{T}(\vartheta)=\mathbb E_{(x,y^\star)\sim D_t^{\mathrm{train}}}\left[\frac{1}{|y^\star|}\sum_{k=1}^{|y^\star|}D_{\mathrm{KL}}\bigl(p_{t,k}^{T}\parallel p_{t,k}^{S}\bigr)\right]`}
        >
          Full-vocabulary forward KL on the current task, computed in FP32. Gradients flow only through the student.
        </Eq>
        <Eq
          name="Anchor views"
          tex={tx`p_{k}^{C}=\pi_{\bar\theta_{t-1}}\bigl(\cdot\mid u,\,\hat c_{<k}\bigr),\qquad p_{k}^{0}=\pi_{\theta_0}\bigl(\cdot\mid u,\,\hat c_{<k}\bigr),\qquad p_{k}^{S}=\pi_\vartheta\bigl(\cdot\mid u,\,\hat c_{<k}\bigr)`}
        >
          For a probe prompt <Tex tex="u" />, <Tex tex={tx`\hat c`} /> is the anchor&rsquo;s own first 32 greedy
          tokens: <Tex tex={tx`\bar\theta_{t-1}`} />&rsquo;s for historical probes and{' '}
          <Tex tex={tx`\theta_0`} />&rsquo;s for general probes.
        </Eq>
        <Eq
          name="Worst-group drift"
          tex={tx`\mathcal D_t^{C}(\vartheta)=\max_{g}\ \frac{1}{|V_g|}\sum_{u\in V_g}\overline{\mathrm{KL}}_u\bigl(p^{C}\parallel p^{S}\bigr),\qquad \mathcal D_t^{G}(\vartheta)=\max_{g}\ \frac{1}{|G_g|}\sum_{u\in G_g}\overline{\mathrm{KL}}_u\bigl(p^{0}\parallel p^{S}\bigr)`}
        >
          <Tex tex={tx`V_g`} /> are the historical probes of task <Tex tex="g" /> in{' '}
          <Tex tex={tx`\mathcal R_{t-1}^{\mathrm{task}}`} />, <Tex tex={tx`G_g`} /> the general probes of source{' '}
          <Tex tex="g" />, and <Tex tex={tx`\overline{\mathrm{KL}}_u`} /> the per-token mean over the completion.
        </Eq>
        <Eq
          name="Constrained problem"
          tex={tx`\vartheta_t^\star\in\arg\min_{\vartheta}\ \mathcal L_t^{T}(\vartheta)\quad\text{s.t.}\quad \mathcal D_t^{C}(\vartheta)\le\epsilon_C,\ \ \mathcal D_t^{G}(\vartheta)\le\epsilon_G`}
        >
          Optimization initializes <Tex tex={tx`\vartheta`} /> from <Tex tex={tx`\bar\theta_{t-1}`} />, where both
          drifts are zero.
        </Eq>
        <Eq
          name="Lagrangian update"
          tex={tx`\mathcal L_t^{\lambda}(\vartheta)=\mathcal L_t^{T}(\vartheta)+\frac{\lambda_C}{\epsilon_C}\,\mathcal D_t^{C}(\vartheta)+\frac{\lambda_G}{\epsilon_G}\,\mathcal D_t^{G}(\vartheta)`}
        >
          Each update descends this objective; a drift term is included only while its multiplier is positive.
        </Eq>
        <Eq
          name="Bounded dual ascent"
          tex={tx`\lambda_j\leftarrow\max\Bigl(0,\ \lambda_j+\eta\,\min\bigl(1,\ \mathcal D_t^{j}/\epsilon_j-1\bigr)\Bigr),\qquad j\in\{C,G\},\quad \eta=0.5`}
        >
          Multipliers start at zero in every stage, and a violation counts at most once per update.
        </Eq>
      </MathDetails>
    </div>
  )
}
