import { vars } from '../lib/css.ts'

export function Abstract() {
  return (
    <section className="section" id="abstract">
      <div className="wrap">
        <h2 className="section-title reveal">Abstract</h2>
        <p className="abstract-text reveal">
          Continual learning requires a model to keep acquiring new skills and knowledge without compromising its
          existing capabilities. Parametric methods suffer from catastrophic forgetting and loss of plasticity, and are
          insensitive to critical samples. Non-parametric methods, such as prompts, skills, and harnesses, leverage
          in-context learning and dominate in practice, but are bounded by context length and lossy plain-text
          compression. We propose HERD (Harness-Evolving Recursive Distillation), a hybrid continual learning system
          that learns quickly in an{' '}
          <mark style={vars({ '--mark': 'var(--harness-soft)' })}>inspectable, bounded harness</mark> and recursively
          distills it into a{' '}
          <mark style={vars({ '--mark': 'var(--predictor-soft)' })}>standalone predictor</mark>. On the harness side,
          the frozen base model writes a skill for each task from a small sample of verified examples and the current
          predictor&rsquo;s mistakes, while examples that the skills leave poorly explained, identified by high
          residuals, are retained in a bounded memory. On the internalization side, the frozen previous checkpoint,
          given the routed skill and the verified answer, teaches its successor through{' '}
          <mark style={vars({ '--mark': 'var(--optim-soft)' })}>full-distribution distillation</mark> under{' '}
          <mark style={vars({ '--mark': 'var(--guard-soft)' })}>two drift budgets</mark> enforced by Lagrange
          multipliers: the previous checkpoint anchors earlier tasks, and the immutable base model anchors general
          capabilities. The resulting predictor{' '}
          <mark style={vars({ '--mark': 'var(--deploy-soft)' })}>needs no harness at inference</mark>. On the 25-stage
          TRACE+17 stream with Qwen3.5-4B, under fixed memory and compute budgets, HERD matches the strongest
          parametric baseline in new-task acquisition while showing the least forgetting among weight-updating methods
          and the strongest retention of general capabilities.
        </p>
      </div>
    </section>
  )
}
