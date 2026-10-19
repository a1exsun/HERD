import { Eq, MathDetails } from '../components/MathDetails.tsx'
import { Tex } from '../components/Tex.tsx'
import { GuardedEvolution } from '../demos/GuardedEvolution.tsx'
import { ResidualRetention } from '../demos/ResidualRetention.tsx'
import { tx } from '../lib/tex.ts'

export function HarnessEvolving() {
  return (
    <div className="subsection" id="harness-evolving">
      <h3 className="reveal">Harness Evolution</h3>
      <div className="text reveal">
        <p>
          We use harness in the sense of DeepSeek Harness (DSH) and of harness-level continual learning: the state and
          procedures around a model at inference, such as system prompts, skills, memories and tools, as opposed to its
          weights. The harness of HERD is <Tex tex={tx`H=(K,\mathcal R)`} />. A skill in the registry{' '}
          <Tex tex="K" /> is an identifier, a routing prefix and instructions. A prompt is routed to the skill with the
          longest prefix that matches it after NFKC normalization, whitespace collapsing and lowercasing; the
          instructions become a fixed system message, and a prompt with no match runs bare. The memory{' '}
          <Tex tex={tx`\mathcal R`} /> holds whole training records, prompt and verified answer, kept verbatim.
        </p>
      </div>

      <div className="para">
        <h4 className="reveal">Proposal</h4>
        <div className="text reveal">
          <p>
            The harness reads the task through a bounded window{' '}
            <Tex tex={tx`A_t\subset D_t^{\mathrm{train}}`} /> of whole records, filled in a seeded order to a budget{' '}
            <Tex tex="B_A=8{,}192" /> tokens: it must decide what to change and what to keep. The frozen checkpoint{' '}
            <Tex tex={tx`\bar\theta_{t-1}`} /> first answers <Tex tex="A_t" /> greedily without a skill; at most four
            wrong answers go to the skill writer, so the skill targets what the model now gets wrong rather than
            restating the task. The base <Tex tex={tx`\theta_0`} /> then writes the skill in one greedy call from up to
            16 records of the window, those whose answers open differently first, and the mistakes, under four
            headers: Format, Rules, Avoid and Examples. Keeping the writer at <Tex tex={tx`\theta_0`} /> keeps it fixed
            across the recursion.
          </p>
          <p>
            The skill&rsquo;s prefix is the longest common prefix of the task&rsquo;s training prompts; its instructions
            are clipped to 768 tokens, and it replaces any skill with the same prefix or identifier.
          </p>
        </div>
      </div>

      <div className="para">
        <h4 className="reveal">Guard</h4>
        <div className="text reveal">
          <p>
            The residual of a record under a model executing a registry is the mean negative log-likelihood of its
            verified answer with the skill, but not the answer, in context. A candidate registry{' '}
            <Tex tex="K'" /> is committed only if the mean residual of <Tex tex="A_t" /> under{' '}
            <Tex tex={tx`\bar\theta_{t-1}`} /> is lower than under <Tex tex={tx`K_{t-1}`} /> and the serialized
            registry fits <Tex tex="B_K=24{,}576" /> tokens; otherwise <Tex tex="K_t=K_{t-1}" />. A skill survives
            only if it helps the checkpoint predict the task.
          </p>
        </div>
        <GuardedEvolution />
      </div>

      <div className="para">
        <h4 className="reveal">Residual refresh</h4>
        <div className="text reveal">
          <p>
            After distillation and merge, the residual of every record in{' '}
            <Tex tex={tx`\mathcal R_{t-1}\cup A_t`} /> is recomputed under <Tex tex={tx`\bar\theta_t`} /> executing{' '}
            <Tex tex="K_t" />, and <Tex tex={tx`\mathcal R_t`} /> keeps whole records in decreasing residual until{' '}
            <Tex tex="B_R" /> is filled. What the skill and the new weights together still cannot predict stays in
            plain text; with one global budget, a record the weights have just absorbed yields its place at once.
          </p>
          <p>
            <Tex tex={tx`\mathcal R`} /> is never replayed and never a training target. It supplies the probes of the
            next stage&rsquo;s checkpoint-drift constraint: on the records the new checkpoint finds hardest, the student
            may not leave it. The memory is what the harness keeps; the skill is what it changes. The memory has{' '}
            <Tex tex="B_R=32{,}768" /> tokens, of which <Tex tex="B_G=8{,}192" /> are reserved for <Tex tex="G" />, so
            task records occupy at most 24,576.
          </p>
        </div>
        <ResidualRetention />
      </div>

      <MathDetails title="Equations for Harness Evolution" count={3}>
        <Eq
          name="Residual"
          tex={tx`r(x,y^*;\theta,K) = -\frac{1}{|y^*|}\sum_{k=1}^{|y^*|}\log p(y^*_k \mid K(x),x,y^*_{<k};\theta)`}
        >
          Scored with only the routed skill and bare prompt in the prefill, conditioning on{' '}
          <Tex tex={tx`y^*_{<k}`} /> at each answer position.
        </Eq>
        <Eq
          name="Guard"
          tex={tx`\bar r_t(K)=\frac{1}{|A_t|}\sum_{(x,y^*)\in A_t} r(x,y^*;\bar\theta_{t-1},K),\qquad K_t=\begin{cases}K' & \text{if } \bar r_t(K')<\bar r_t(K_{t-1})\ \text{and}\ \mathrm{size}(K')\le B_K\\ K_{t-1} & \text{otherwise}\end{cases}`}
        >
          A candidate is kept only when the skill alone, without a replayed answer in the prefill, makes the window
          easier to predict.
        </Eq>
        <Eq
          name="Memory refresh"
          tex={tx`\mathcal R^{(i)}=\begin{cases}\mathcal R^{(i-1)}\cup\{\rho_{(i)}\} & \text{if } \tau_{i-1}+|\rho_{(i)}|\le B_R-B_G\\ \mathcal R^{(i-1)} & \text{otherwise}\end{cases}`}
        >
          <Tex tex={tx`\rho_{(1)},\ldots,\rho_{(n)}`} /> are the records of <Tex tex={tx`\mathcal R_{t-1}\cup A_t`} />,
          deduplicated by identifier, in decreasing <Tex tex={tx`r(\cdot;\bar\theta_t,K_t)`} /> with ties broken by
          identifier; <Tex tex={tx`\tau_i`} /> is the token count of <Tex tex={tx`\mathcal R^{(i)}`} />. One scan from{' '}
          <Tex tex={tx`\mathcal R^{(0)}=\varnothing`} /> gives <Tex tex={tx`\mathcal R_t=\mathcal R^{(n)}`} />: a
          record enters only if it fits whole, and a smaller record further down may still enter after a larger one is
          skipped.
        </Eq>
      </MathDetails>
    </div>
  )
}
