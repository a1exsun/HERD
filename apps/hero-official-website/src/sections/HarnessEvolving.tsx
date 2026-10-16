import { Eq, MathDetails } from '../components/MathDetails.tsx'
import { Tex } from '../components/Tex.tsx'
import { GuardedEvolution } from '../demos/GuardedEvolution.tsx'
import { ResidualRetention } from '../demos/ResidualRetention.tsx'
import { tx } from '../lib/tex.ts'

export function HarnessEvolving() {
  return (
    <div className="subsection" id="harness-evolving">
      <h3 className="reveal">Harness Evolving</h3>
      <div className="text reveal">
        <p>
          The skill registry <Tex tex="K_t" /> holds one skill per task. A skill has a routing prefix, the longest
          common prefix of the task&rsquo;s training prompts cut back to a whitespace boundary, and instructions on
          the task&rsquo;s output format, decision rules, typical errors, and short worked examples. A new skill
          replaces any skill with the same prefix or task. The residual component decomposes as{' '}
          <Tex tex={tx`\mathcal R_t=\mathcal G\cup\mathcal R_t^{\mathrm{task}}`} />, where{' '}
          <Tex tex={tx`\mathcal R_t^{\mathrm{task}}`} /> contains retained task examples and <Tex tex={tx`\mathcal G`} />{' '}
          is a fixed set of 48 non-test general probes (GSM8K, IFEval, and SuperNI) reserved before the stream.
          Neither component enters ordinary harness execution. Task residuals supply the probes for checkpoint drift,
          and general probes supply the probes for base drift.
        </p>
      </div>

      <div className="para">
        <h4 className="reveal">Guarded evolution</h4>
        <div className="text reveal">
          <p>
            A per-stage seeded shuffle of the task&rsquo;s training records, kept whole within{' '}
            <Tex tex="B_A=8{,}192" /> tokens, forms the bounded evidence <Tex tex="A_t" />. The frozen previous
            checkpoint <Tex tex={tx`\bar\theta_{t-1}`} /> answers <Tex tex="A_t" /> greedily, and up to four answers
            that differ from the verified output become mistakes. The frozen base model <Tex tex={tx`\theta_0`} /> then
            writes the skill in one greedy call with thinking off. It sees up to 16 candidates from{' '}
            <Tex tex="A_t" />, those whose answers open differently first so that a task with several output
            conventions shows several, together with the mistakes, and answers under four headers: Format, Rules,
            Avoid (grounded in the mistakes), and Examples. The instructions are clipped to 768 tokens.
          </p>
          <p>
            The frozen teacher is the skill&rsquo;s only consumer during training, so the guard measures exactly what
            distillation reads and needs no generation. The candidate <Tex tex="K'_t" /> is committed unless its
            instructions are empty, the serialized registry exceeds <Tex tex="B_K=24{,}576" /> tokens, or it raises the
            teacher&rsquo;s mean negative log-likelihood of the verified answers in <Tex tex="A_t" /> by more than{' '}
            <Tex tex={tx`\tau_K=0.05`} /> nats per token. Because the teacher sees the verified answer replayed, its
            likelihood is already near its maximum, so a gain is rarely measurable; a large rise means the skill
            fights the answer. Otherwise <Tex tex="K_t=K_{t-1}" />, with no retry. Proposal and guard costs count
            toward <Tex tex="B_C" />.
          </p>
        </div>
        <GuardedEvolution />
      </div>

      <div className="para">
        <h4 className="reveal">Residual retention</h4>
        <div className="text reveal">
          <p>
            For each <Tex tex={tx`z=(x,y^\star)`} /> in <Tex tex={tx`\mathcal R_{t-1}^{\mathrm{task}}\cup A_t`} />, the
            compression residual is the mean target negative log-likelihood under the frozen{' '}
            <Tex tex={tx`\bar\theta_{t-1}`} /> executing the committed <Tex tex="K_t" /> on <Tex tex="x" />, with no
            target in context. TopBudget orders records by decreasing residual, breaking ties by stable identifier, and
            scans this order once, adding a record only if its complete serialization fits the remaining budget{' '}
            <Tex tex={tx`B_R-B_G=24{,}576`} /> tokens. Because the scoring model sees the skill but not{' '}
            <Tex tex={tx`\mathcal R_t`} />, examples the skill already explains receive low scores and poorly
            represented examples are retained preferentially. Eviction is irreversible.
          </p>
        </div>
        <ResidualRetention />
      </div>

      <div className="para">
        <h4 className="reveal">Bounded state</h4>
        <div className="text reveal">
          <p>
            One global budget, rather than a separate allowance per task, constrains the complete harness state. HERD
            enforces <Tex tex={tx`\mathrm{Tok}(K_t)\le B_K`} /> and{' '}
            <Tex tex={tx`\mathrm{Tok}(\mathcal R_t)\le B_R=32{,}768`} />, of which <Tex tex="B_G=8{,}192" /> is reserved
            for <Tex tex={tx`\mathcal G`} />, so <Tex tex={tx`\mathrm{Tok}(H_t)\le B_K+B_R=B_M`} />. The number of
            observed tasks affects which examples survive, not the capacity of the state. Like any fixed-capacity
            learner, HERD relies on reusable structure and cannot preserve an arbitrarily long incompressible stream
            losslessly.
          </p>
        </div>
      </div>

      <MathDetails title="Equations for Harness Evolving" count={5}>
        <Eq
          name="Skill proposal"
          tex={tx`K'_t=K_{t-1}\oplus s_t,\qquad s_t=\bigl(\mathrm{prefix}(D_t^{\mathrm{train}}),\ \mathrm{clip}_{768}\,\pi_{\theta_0}^{\mathrm{greedy}}(\cdot\mid A_t,\mathcal M_t)\bigr),\qquad \mathrm{Tok}(A_t)\le B_A`}
        >
          <Tex tex={tx`\mathcal M_t`} /> holds up to four records of <Tex tex="A_t" /> that the frozen{' '}
          <Tex tex={tx`\bar\theta_{t-1}`} /> answers wrongly. <Tex tex={tx`\oplus`} /> replaces a skill with the same
          prefix or task.
        </Eq>
        <Eq
          name="Commit rule"
          tex={tx`K_t=\begin{cases}K'_t & \text{if } s_t\neq\varnothing,\ \mathrm{Tok}(K'_t)\le B_K,\ \mathrm{NLL}_t(K'_t)-\mathrm{NLL}_t(K_{t-1})\le\tau_K\\ K_{t-1} & \text{otherwise}\end{cases}`}
        >
          <Tex tex={tx`\mathrm{NLL}_t(K)`} /> is the frozen teacher&rsquo;s mean per-token negative log-likelihood of
          the verified answers in <Tex tex="A_t" />, with the skill routed by <Tex tex="K" /> and the answer replayed
          as in distillation.
        </Eq>
        <Eq
          name="Compression residual"
          tex={tx`r_t(z)=-\frac{1}{|y^\star|}\sum_{k=1}^{|y^\star|}\log\pi_{\bar\theta_{t-1}}\bigl(y_k^\star\mid \Omega_{\mathcal F}(K_t,x),\,x,\,y_{<k}^\star\bigr)`}
        >
          A record whose prompt and answer exceed the training length limit receives{' '}
          <Tex tex={tx`r_t(z)=+\infty`} />.
        </Eq>
        <Eq
          name="Retention"
          tex={tx`\mathcal U_t=\mathrm{Dedup}\bigl(\mathcal R_{t-1}^{\mathrm{task}}\cup A_t\bigr),\qquad \mathcal R_t^{\mathrm{task}}=\mathrm{TopBudget}_{B_R-B_G}(\mathcal U_t;\,r_t)`}
        >
          Ties are broken by stable sample identifier. Records are not truncated during selection, and unused capacity
          remains empty.
        </Eq>
        <Eq
          name="State budget"
          tex={tx`\mathrm{Tok}(K_t)\le B_K,\qquad \mathrm{Tok}(\mathcal R_t)\le B_R,\qquad \mathrm{Tok}(H_t)\le B_K+B_R=B_M`}
        >
          The general probes satisfy <Tex tex={tx`\mathrm{Tok}(\mathcal G)\le B_G`} />.
        </Eq>
      </MathDetails>
    </div>
  )
}
