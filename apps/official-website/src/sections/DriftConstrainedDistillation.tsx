import { Eq, MathDetails } from '../components/MathDetails.tsx'
import { Tex } from '../components/Tex.tsx'
import { LoraMerge } from '../demos/LoraMerge.tsx'
import { PolicyViews } from '../demos/PolicyViews.tsx'
import { tx } from '../lib/tex.ts'

export function DriftConstrainedDistillation() {
  return (
    <div className="subsection" id="drift-constrained-distillation">
      <h3 className="reveal">Drift-Constrained Distillation</h3>

      <div className="para">
        <h4 className="reveal">Teacher and student</h4>
        <div className="text reveal">
          <p>
            The teacher <Tex tex="p^H" /> is the frozen <Tex tex={tx`\bar\theta_{t-1}`} /> under a privileged context:
            the routed skill <Tex tex="K_t(x)" /> as the system message and the verified answer{' '}
            <Tex tex="y^*" /> replayed as the model&rsquo;s own previous turn, followed by the request &ldquo;Repeat your
            previous answer exactly, with no other text.&rdquo; The student <Tex tex="p^S" /> is the same weights with a
            freshly initialized LoRA adapter and sees only the bare prompt. The context carries no wording about the
            answer, so the student learns the answer and not the scaffolding around it.
          </p>
          <p>
            The student minimizes the full-vocabulary forward KL divergence at every position of the verified answer.
            Scoring the verified answer&rsquo;s own prefix rather than student samples closes degenerate paths such as
            echoing the prompt, on which teacher and student agree and the gradient vanishes. Self-distillation of this
            kind shifts the model less than SFT and thus forgets less.
          </p>
        </div>
        <PolicyViews />
      </div>

      <div className="para">
        <h4 className="reveal">Drift</h4>
        <div className="text reveal">
          <p>
            A probe is a prompt with the first <Tex tex="n\le 32" /> greedy tokens of an anchor; drift is the worst
            probe group&rsquo;s mean KL from the anchor to the student. Only the opening is scored because it decides
            the mode of a response: a wrong option or a dropped choice letter is a flip of the first few tokens that a
            long average would hide. The worst group is taken because a collapse confined to one task, such as
            Seq-LoRA&rsquo;s correct ScienceQA answers falling from 94 to 15, barely moves a mean over all probes,
            which is also why <Tex tex={tx`\Delta C_{\min}`} /> is reported.
          </p>
        </div>
      </div>

      <div className="para">
        <h4 className="reveal">Two anchors</h4>
        <div className="text reveal">
          <p>
            Checkpoint drift <Tex tex="D^C" /> anchors <Tex tex={tx`\bar\theta_{t-1}`} /> on probes from{' '}
            <Tex tex={tx`\mathcal R_{t-1}`} />, up to eight per retained task, grouped by task, and bounds per-stage
            drift and the recently learned tasks. Base drift <Tex tex="D^0" /> anchors <Tex tex={tx`\theta_0`} /> on
            the 48 records of <Tex tex="G" />, non-test GSM8K, IFEval and Super-NaturalInstructions records disjoint
            from the G-test, grouped by source, and bounds cumulative drift and general capability. Each bounds what
            the other cannot: the previous checkpoint keeps every step small, but small steps accumulate; the base
            cannot protect a task it never knew. The budgets <Tex tex={tx`\varepsilon_C=\varepsilon_0=0.05`} /> are
            in units of divergence, so one pair serves every stage, which a penalty coefficient could not since the
            scale of the loss changes from stage to stage.
          </p>
        </div>
      </div>

      <div className="para">
        <h4 className="reveal">Enforcement</h4>
        <div className="text reveal">
          <p>
            The stage minimizes the distillation loss over the adapter parameters <Tex tex={tx`\phi`} /> subject to
            both budgets, with multipliers <Tex tex={tx`\lambda_C,\lambda_0\ge 0`} />. Each update measures both
            drifts, adds every active constraint&rsquo;s worst-group divergence to the loss with weight{' '}
            <Tex tex={tx`\lambda_j/\varepsilon_j`} />, steps on <Tex tex={tx`\phi`} /> only, and moves the multipliers
            by bounded dual ascent: within budget a multiplier decays to zero and leaves learning alone, and over
            budget it grows and pulls the student back. The final adapter&rsquo;s drift is audited against the
            original budgets.
          </p>
          <p>
            A stage runs 32 updates of 32 records with AdamW without weight decay, a learning rate of{' '}
            <Tex tex="10^{-4}" /> decaying linearly to zero, and gradient-norm clipping at 1.0; the final iterate is
            kept. The student adds LoRA factors of rank 32, <Tex tex={tx`\alpha=64`} />, and no dropout to every linear
            layer of the text decoder.
          </p>
        </div>
      </div>

      <div className="para">
        <h4 className="reveal">One checkpoint, three roles</h4>
        <div className="text reveal">
          <p>
            One copy of the previous checkpoint is resident as the student&rsquo;s backbone, the teacher{' '}
            <Tex tex="p^H" />, its forward pass with the adapter disabled, and the anchor <Tex tex="p^C" />; both
            anchors&rsquo; states on their fixed probes are cached. The adapter is cast to bf16 and merged into the
            dense weights, which become <Tex tex={tx`\bar\theta_t`} /> and the backbone of the next stage&rsquo;s fresh
            adapter, so every stage starts from <Tex tex={tx`\Delta W=0`} />. With a fixed number of updates and
            probes per stage, the cost does not grow with <Tex tex="t" />; only <Tex tex={tx`\bar\theta_t`} /> is
            deployed.
          </p>
        </div>
        <LoraMerge />
      </div>

      <MathDetails title="Equations for Drift-Constrained Distillation" count={5}>
        <Eq
          name="Distillation objective"
          tex={tx`\mathcal{L}_{\mathrm{dist}} = \mathbb{E}_{(x,y^*)}\,\frac{1}{|y^*|}\sum_{k=1}^{|y^*|}\mathrm{KL}\!\left(p^H_k \,\|\, p^S_k\right)`}
        >
          With <Tex tex={tx`p^H_k=p^H(\cdot\mid K_t(x),x,y^*,y^*_{<k})`} /> and{' '}
          <Tex tex={tx`p^S_k=p^S(\cdot\mid x,y^*_{<k})`} />, aligned by answer-token index.
        </Eq>
        <Eq
          name="Drift"
          tex={tx`D(P;a) = \max_{g}\,\mathbb{E}_{P_g}\,\frac{1}{n}\sum_{k=1}^{n}\mathrm{KL}(p^{a}_k \,\|\, p^S_k)`}
        >
          A probe is a prompt <Tex tex="x" /> with the first <Tex tex="n" /> greedy tokens <Tex tex={tx`\hat y`} /> of
          an anchor <Tex tex="a" />, with <Tex tex={tx`p^a_k=p^a(\cdot\mid x,\hat y_{<k})`} />;{' '}
          <Tex tex={tx`P_g`} /> are the probe groups.
        </Eq>
        <Eq
          name="Constrained problem"
          tex={tx`\min_{\phi}\;\mathcal{L}_{\mathrm{dist}}(\phi)\quad\text{s.t.}\quad D^C\le\varepsilon_C,\; D^0\le\varepsilon_0`}
        >
          <Tex tex="D^C" /> anchors <Tex tex={tx`\bar\theta_{t-1}`} /> on probes from{' '}
          <Tex tex={tx`\mathcal R_{t-1}`} />; <Tex tex="D^0" /> anchors <Tex tex={tx`\theta_0`} /> on{' '}
          <Tex tex="G" />.
        </Eq>
        <Eq
          name="Update"
          tex={tx`\mathcal{L}(\phi)=\mathcal{L}_{\mathrm{dist}}(\phi)+\sum_{j\in\{C,0\}}[\lambda_j>0]\,\frac{\lambda_j}{\varepsilon_j}\,\bar D^{j}_{g_j^*}(\phi)`}
        >
          <Tex tex={tx`g_j^*`} /> is the worst group of drift <Tex tex="j" /> at that measurement and{' '}
          <Tex tex={tx`\bar D^j_{g}`} /> that group&rsquo;s mean divergence.
        </Eq>
        <Eq
          name="Bounded dual ascent"
          tex={tx`\lambda_j \leftarrow \max(0,\,\lambda_j + \eta\min(1,\,D^j/\varepsilon_j - 1)),\qquad \eta=0.5`}
        >
          Both multipliers start at zero.
        </Eq>
      </MathDetails>
    </div>
  )
}
