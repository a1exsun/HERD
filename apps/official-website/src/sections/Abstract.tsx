import { vars } from '../lib/css.ts'

export function Abstract() {
  return (
    <section className="section" id="abstract">
      <div className="wrap">
        <h2 className="section-title reveal">Abstract</h2>
        <p className="abstract-text reveal">
          Continual learning requires a model to keep acquiring new skills and knowledge without compromising its
          existing capabilities. Parametric methods suffer from catastrophic forgetting and loss of plasticity.
          Non-parametric methods, such as prompts, skills, and harnesses, leverage in-context learning but are limited by
          context length and lossy plain-text compression. To effectively combine their complementary strengths, we
          propose HERD (Harness-Evolving Recursive Distillation), a hybrid continual learning framework that{' '}
          <mark style={vars({ '--mark': 'var(--harness-soft)' })}>learns efficiently in the harness</mark> and{' '}
          <mark style={vars({ '--mark': 'var(--predictor-soft)' })}>internalizes experience into cumulative
          capabilities</mark> via distillation. On the harness side, HERD continually refines its skills to capture
          reusable patterns, while retaining high-residual examples that the skills fail to capture in a bounded
          memory. On the parameter side, HERD performs{' '}
          <mark style={vars({ '--mark': 'var(--optim-soft)' })}>drift-constrained distillation</mark> with{' '}
          <mark style={vars({ '--mark': 'var(--guard-soft)' })}>dual anchors</mark>, substantially reducing forgetting
          over long recursive runs. On the 25-stage task stream of TRACE+17, under fixed memory and compute budgets,
          HERD matches the strongest parametric baseline in new-task acquisition while exhibiting the least forgetting
          and the strongest retention of general capabilities.
        </p>
        {/* Same asset as the README; GitHub redirects it to a fresh signed URL on every request. */}
        <video
          className="fig abstract-video reveal"
          src="https://github.com/user-attachments/assets/48121e37-5e83-4594-b6e9-8bad2563087c"
          poster="video-poster.jpg"
          controls
          playsInline
          preload="metadata"
          aria-label="HERD overview video"
        />
      </div>
    </section>
  )
}
