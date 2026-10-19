import katex from 'katex'
import { useMemo } from 'react'

interface TexProps {
  tex: string
  block?: boolean
}

/** Typesets a TeX string with KaTeX. Everything is bundled, so formulas render offline. */
export function Tex({ tex, block = false }: TexProps) {
  const html = useMemo(
    () =>
      katex.renderToString(tex, {
        displayMode: block,
        throwOnError: false,
        strict: 'ignore',
      }),
    [tex, block],
  )
  if (block) return <div className="tex-block" dangerouslySetInnerHTML={{ __html: html }} />
  return <span className="tex" dangerouslySetInnerHTML={{ __html: html }} />
}
