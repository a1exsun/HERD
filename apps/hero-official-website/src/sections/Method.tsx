import { Eq, MathDetails } from '../components/MathDetails.tsx'
import { Tex } from '../components/Tex.tsx'
import { Paradigms } from '../demos/Paradigms.tsx'
import { StageAnatomy } from '../demos/StageAnatomy.tsx'
import { tx } from '../lib/tex.ts'
import { DriftConstrainedDistillation } from './DriftConstrainedDistillation.tsx'
import { HarnessEvolving } from './HarnessEvolving.tsx'

const ROWS: [string, string, string, string][] = [
  ['Adapted state', 'External state', 'Harness and parameters', 'Parameters'],
  ['Carried forward', 'Harness', 'Harness and predictor', 'Predictor, optionally with retained training state'],
  ['Required at inference', 'Model and harness', 'Predictor', 'Predictor'],
  ['Drift control', 'Parameters are frozen', 'Checkpoint and base drift budgets', 'None in standard sequential training'],
]

export function Method() {
  return (
    <section className="section" id="method">
      <div className="wrap">
        <h2 className="section-title reveal">Method</h2>
        <p className="experiment-note">
          This description follows the paper. The current release runner still permits a 0.05 rise in
          replay-conditioned teacher NLL and selects residuals before training; the paper uses strict residual
          improvement without answer replay and refreshes memory after merge. The runner is unchanged here.
        </p>

        <div className="subsection" id="overview">
          <h3 className="reveal">Overview</h3>
          <div className="text reveal">
            <p>
              HERD couples a persistent, bounded harness state <Tex tex={tx`H_t=(K_t,\mathcal R_t)`} /> with a
              standalone deployed predictor <Tex tex={tx`\bar\theta_t`} />. The skill registry <Tex tex="K_t" /> holds
              one inspectable skill per task, and the residual memory <Tex tex={tx`\mathcal R_t`} /> holds the verified
              examples those skills explain worst. The predictor recursively internalizes the behavior that the latest
              harness exposes and uses neither component of <Tex tex="H_t" /> at inference.
            </p>
            <p>
              Non-parametric continual learning adapts external state and keeps the model frozen, so the external state
              is required at inference. Conventional parametric continual learning passes the trained model between
              stages, inheriting useful knowledge but also any interference and drift induced by task specialization.
              HERD retains recursive parameter inheritance but bounds every transition by two behavioral drift
              budgets.
            </p>
          </div>
          <Paradigms />
          <div className="ledger-scroll reveal">
            <div className="ledger" style={{ minWidth: 700 }}>
              <div className="ledger-row" data-head="true">
                <span />
                <span>Non-parametric CL</span>
                <span data-hero="true">HERD</span>
                <span>Parametric CL</span>
              </div>
              {ROWS.map(([name, a, b, c]) => (
                <div className="ledger-row" key={name}>
                  <span>{name}</span>
                  <span>{a}</span>
                  <span data-hero="true">{b}</span>
                  <span>{c}</span>
                </div>
              ))}
            </div>
          </div>
          <div className="text reveal">
            <p>
              Stage <Tex tex="t" /> receives <Tex tex={tx`\bar\theta_{t-1}`} /> and <Tex tex="H_{t-1}" /> and runs three
              steps. (1) <em>Harness evolving:</em> the frozen base model <Tex tex={tx`\theta_0`} /> writes the current
              task&rsquo;s skill from bounded evidence <Tex tex="A_t" />, and a residual-improvement guard decides whether it is
              committed. (2) <em>Drift-constrained distillation:</em> the student,{' '}
              <Tex tex={tx`\bar\theta_{t-1}`} /> with a fresh LoRA, learns from the frozen{' '}
              <Tex tex={tx`\bar\theta_{t-1}`} /> given the routed skill and the verified answer; the previous checkpoint
              anchors earlier tasks on retained probes and the immutable base model <Tex tex={tx`\theta_0`} /> anchors
              general capabilities, each under a budget held by a Lagrange multiplier. (3) <em>Merge and refresh:</em> the LoRA is
              merged into <Tex tex={tx`\bar\theta_t`} />, then residuals are recomputed under the merged weights and
              <Tex tex="K_t" /> to retain the hardest examples within the memory budget. The predictor deploys alone
              and initializes stage <Tex tex="t+1" />.
            </p>
            <p>
              Only <Tex tex="K_t" /> evolves; the framework that executes it is fixed. A prompt runs only the skill
              whose routing prefix is the longest match, shown as the system message. The registry uses the{' '}
              <code>hero-harness-spec-v1</code> format, which the stream runner and the DeepSeek Harness (DSH) plugin
              execute identically. HERD is harness-agnostic in principle and applies to any harness that satisfies the
              functional contract <Tex tex={tx`\Omega_{\mathcal F}(K_t,x)`} />.
            </p>
          </div>
          <StageAnatomy />
          <MathDetails title="Notation" count={5}>
            <Eq
              name="Task stream"
              tex={tx`\mathcal S=(\mathcal T_1,\ldots,\mathcal T_T),\qquad \mathcal T_t=\bigl(D_t^{\mathrm{train}},D_t^{\mathrm{val}},D_t^{\mathrm{test}},m_t\bigr)`}
            >
              Tasks arrive in a fixed order. Each task has disjoint splits and a metric <Tex tex="m_t" /> normalized to{' '}
              <Tex tex="[0,1]" />.
            </Eq>
            <Eq
              name="Learner update"
              tex={tx`\mathcal L_t=A(\mathcal L_{t-1},E_t),\qquad \mathcal L_t=(f_t,M_t),\quad M_t=(H_t,\mathcal G)`}
            >
              Any information carried across stages must be represented in <Tex tex={tx`\mathcal L_{t-1}`} />. In HERD
              the retained state includes the harness and the fixed general probes.
            </Eq>
            <Eq
              name="Budgets"
              tex={tx`\mathrm{size}(M_t)\le B_M,\qquad \sum_{t=1}^{T}\mathrm{cost}(A,\mathcal L_{t-1},E_t)\le B_C`}
            >
              Storage accounting includes every historical artifact available to the learner. Proposals, guard
              evaluations, probe decoding, and rejected candidates count toward <Tex tex="B_C" />.
            </Eq>
            <Eq
              name="Harness contract"
              tex={tx`\Omega_{\mathcal F}(K_t,x)=\mathsf{Render}\bigl(\mathsf{Route}(K_t,x)\bigr),\qquad \mathsf{Route}(K_t,x)=\arg\max_{s\in K_t:\ \nu(x)\,\succeq\,\nu(s.\mathrm{prefix})}\bigl|\nu(s.\mathrm{prefix})\bigr|`}
            >
              <Tex tex={tx`\nu`} /> applies NFKC normalization, collapses whitespace and lowercases, and{' '}
              <Tex tex={tx`\succeq`} /> means &ldquo;starts with&rdquo;. <Tex tex={tx`\mathsf{Render}`} /> places the
              routed skill in the system message; an unmatched prompt runs without a skill. <Tex tex="K_t" /> evolves,
              and <Tex tex={tx`\mathcal F`} /> remains fixed.
            </Eq>
            <Eq
              name="Evaluation"
              tex={tx`\mathrm{AA}=\frac1T\sum_{j=1}^{T}R_{T,j},\quad \mathrm F=\frac{1}{T-1}\sum_{j=1}^{T-1}\max\bigl(0,R_{j,j}-R_{T,j}\bigr),\quad \Delta C=\frac1K\sum_{k=1}^{K}\bigl(C_{T,k}-C_{0,k}\bigr)`}
            >
              Final average performance, forgetting, and general-capability change are reported separately.
            </Eq>
          </MathDetails>
        </div>

        <HarnessEvolving />
        <DriftConstrainedDistillation />
      </div>
    </section>
  )
}
