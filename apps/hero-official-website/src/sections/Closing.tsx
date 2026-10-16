import { useRef, useState } from 'react'

const BIBTEX = `@misc{sun2026herd,
  title  = {HERD: Harness-Evolving Recursive Distillation for Continual Learning},
  author = {Sun, Xun and Vu, Trang and Nguyen, Thanh Thi},
  year   = {2026},
  note   = {Under review}
}`

export function Cite() {
  const pre = useRef<HTMLPreElement>(null)
  const [copied, setCopied] = useState(false)

  const selectText = () => {
    const el = pre.current
    const selection = window.getSelection()
    if (!el || !selection) return
    const range = document.createRange()
    range.selectNodeContents(el)
    selection.removeAllRanges()
    selection.addRange(range)
  }

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(BIBTEX)
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1800)
    } catch {
      selectText()
    }
  }

  return (
    <div className="cite" id="cite">
      <div className="wrap">
        <span className="cite-label">BibTeX</span>
        <div className="bib">
          <pre ref={pre}>{BIBTEX}</pre>
          <button type="button" className="action" onClick={copy}>
            {copied ? 'Copied' : 'Copy'}
          </button>
        </div>
      </div>
    </div>
  )
}

export function Footer() {
  return (
    <footer className="footer">
      <div className="wrap">
        <span>Xun Sun, Trang Vu, Thanh Thi Nguyen · HERD: Harness-Evolving Recursive Distillation for Continual Learning</span>
        <span>Figures marked “Schematic” use illustrative values and do not report experimental results.</span>
      </div>
    </footer>
  )
}
