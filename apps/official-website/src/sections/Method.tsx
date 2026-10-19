import { Eq, MathDetails } from '../components/MathDetails.tsx'
import { Tex } from '../components/Tex.tsx'
import { Paradigms } from '../demos/Paradigms.tsx'
import { StageAnatomy } from '../demos/StageAnatomy.tsx'
import { WorkedExample } from '../demos/WorkedExample.tsx'
import { tx } from '../lib/tex.ts'
import { DriftConstrainedDistillation } from './DriftConstrainedDistillation.tsx'
import { HarnessEvolving } from './HarnessEvolving.tsx'

const ROWS: [string, string, string, string][] = [
  ['Where experience is kept', 'External memory, context or harness', 'Harness and weights', 'Weights'],
  ['Carried forward', 'Harness', 'Harness and weights', 'Weights, optionally with replay data'],
  ['Required at inference', 'Model and harness', 'Weights only', 'Weights'],
  ['Main limit', 'Bounded, lossy plain-text context', 'Drift, bounded by two anchors', 'Forgetting and loss of plasticity'],
]

export function Method() {
  return (
    <section className="section" id="method">
      <div className="wrap">
        <h2 className="section-title reveal">Method</h2>

        <div className="subsection" id="overview">
          <h3 className="reveal">Overview</h3>
          <div className="text reveal">
            <p>
              Parametric continual learning suffers from catastrophic forgetting and loss of plasticity, and gives
              little control over what is internalized: raw experience carries noise, duplicates and conflicts alongside
              the few samples that matter. A harness of prompts, skills and memories avoids both, but nothing
              accumulates in the model itself. Whatever it acts on must pass through the context, so experience is
              compressed into bounded plain text, and the more experience accumulates, the more is lost.
            </p>
            <p>
              HERD therefore lets each layer store what it is best suited for: <em>capacity in the weights, selection
              in the harness</em>. The harness is the fast, selective learning layer and the weights are the slow,
              persistent layer, much as episodic memory formed during the day is consolidated into long-term memory
              during sleep. After every stage HERD internalizes its harness into the weights, under fixed memory and
              compute budgets and at no added inference cost. The harness is kept, inspectable and editable, for the
              next stage; only the weights are deployed.
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
              HERD is designed around recursion: it must run stably stage after stage, which raises two challenges.
              First, the context is bounded, so the harness must decide what to change and what to keep. Second,
              recursive training drifts, and unchecked drift ends in collapse. HERD answers each with one module.{' '}
              <em>Harness evolution:</em> the model iterates on its skills to abstract reusable rules, and retains in a
              bounded memory the high-residual samples the skills cannot cover. <em>Drift-constrained distillation
              with dual anchors:</em> the student is the previous checkpoint trained to match itself under the harness;
              the previous checkpoint bounds per-stage drift and the base model bounds cumulative drift.
            </p>
            <p>
              HERD after stage <Tex tex="t" /> is <Tex tex={tx`\mathcal L_t=(f_{\bar\theta_t},(K_t,\mathcal R_t,G))`} />:
              merged dense weights <Tex tex={tx`\bar\theta_t`} />, the harness{' '}
              <Tex tex={tx`H_t=(K_t,\mathcal R_t)`} />, and a fixed set <Tex tex="G" /> of general probes reserved
              before the stream. One stage learns the task twice. First, the harness evolves on a bounded window of the
              task. Second, the checkpoint, given the skill and the verified answer, teaches a fresh adapter that sees
              only the bare prompt, under drift budgets toward <Tex tex={tx`\bar\theta_{t-1}`} /> and{' '}
              <Tex tex={tx`\theta_0`} />; the adapter is merged into <Tex tex={tx`\bar\theta_t`} />, the memory is
              refreshed under the merged weights, and the harness is carried on. One quantity, the residual of a
              record, runs through all of it.
            </p>
          </div>
          <StageAnatomy />
          <WorkedExample />
          <MathDetails title="Preliminaries" count={5}>
            <Eq
              name="Task stream"
              tex={tx`\mathcal S=(\mathcal T_1,\ldots,\mathcal T_T),\qquad E_t=(D_t^{\mathrm{train}},D_t^{\mathrm{val}})`}
            >
              An ordered stream of tasks, each with its own splits and a metric <Tex tex="m_t" /> in{' '}
              <Tex tex="[0,1]" />. Every example is an input and a target text; any dataset-specific instruction is
              part of the example.
            </Eq>
            <Eq
              name="Learner update"
              tex={tx`\mathcal L_t=A(\mathcal L_{t-1},E_t),\qquad \mathcal L_t=(f_t,M_t),\qquad \mathcal L_0=(f_{\theta_0},M_0)`}
            >
              Stage <Tex tex="t" /> exposes <Tex tex="E_t" /> once; earlier training sets are reachable only through{' '}
              <Tex tex="M" />, which holds everything the learner retains across stages, external indexes included.
            </Eq>
            <Eq
              name="Budgets"
              tex={tx`\mathrm{size}(M_t)\le B_M,\qquad \sum_{t}\mathrm{cost}(A,\mathcal L_{t-1},E_t)\le B_C`}
            >
              Retained state and update compute are bounded, and any context attached at inference is reported with
              the scores.
            </Eq>
            <Eq
              name="Harness"
              tex={tx`H=(K,\mathcal R),\qquad B_M=B_K+B_R,\qquad f(x;\theta,K)`}
            >
              <Tex tex="K" /> is a registry of skills, each a routing prefix and instructions; an input{' '}
              <Tex tex="x" /> is routed by prefix, and the matched instructions <Tex tex="K(x)" /> become the system
              message. <Tex tex={tx`\mathcal R`} /> is a residual memory of whole training records, prompt and verified
              answer, kept verbatim. <Tex tex={tx`f(x;\theta,K)`} /> is the model <Tex tex={tx`\theta`} /> answering{' '}
              <Tex tex="x" /> under the system message <Tex tex="K(x)" />.
            </Eq>
            <Eq
              name="Metrics"
              tex={tx`\mathrm{AA}=\tfrac{1}{T}\textstyle\sum_{j}R_{T,j},\quad \mathrm F=\tfrac{1}{T-1}\textstyle\sum_{j<T}\max(0,\,R_{j,j}-R_{T,j}),\quad \Delta C_{\min}=\min_{k}\Delta C_k`}
            >
              <Tex tex={tx`R_{t,j}`} /> is the score on task <Tex tex="j" /> after stage <Tex tex="t" /> and{' '}
              <Tex tex={tx`\Delta C_k=C_{T,k}-C_{0,k}`} /> the change on general probe <Tex tex="k" />; the mean change{' '}
              <Tex tex={tx`\Delta C`} /> is also reported. The objective is to maximize AA and minimize F without
              negative <Tex tex={tx`\Delta C`} /> or <Tex tex={tx`\Delta C_{\min}`} />, within <Tex tex="B_M" /> and{' '}
              <Tex tex="B_C" />.
            </Eq>
          </MathDetails>
        </div>

        <HarnessEvolving />
        <DriftConstrainedDistillation />
      </div>
    </section>
  )
}
