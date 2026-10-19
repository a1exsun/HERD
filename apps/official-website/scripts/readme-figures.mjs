// Exports the site's animated figures as self-contained SVG files for the repository README.
//
// GitHub renders README images through <img>, which allows embedded CSS animation but no scripts.
// Each figure is therefore rendered once per scene with the site's own React components (through
// Vite SSR, with the carousel hook replaced by a fixed scene index), and the scenes are compiled
// into one SVG in which every element is shown during the scenes it belongs to.
//
//   pnpm --filter @hero/official-website readme:figures

import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { createServer } from 'vite'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const outDir = resolve(root, '../../public/readme')

/* ------------------------------------------------------------------ theme */

const THEMES = {
  light: {
    surface: '#ffffff', ink: '#12151c', 'ink-2': '#394150', muted: '#667085', line: '#dce0e8',
    'line-strong': '#bfc6d2', grid: 'rgba(18,21,28,0.045)',
    harness: '#d2741a', 'harness-soft': '#fbeedd', predictor: '#2f66b3', 'predictor-soft': '#e6eefa',
    optim: '#c4473f', 'optim-soft': '#fae8e6', guard: '#7548a8', 'guard-soft': '#f0e9f8',
    deploy: '#2a8552', 'deploy-soft': '#e3f3ea', aux: '#8a93a3', 'aux-soft': '#eff1f5',
    'viz-hero': '#1baf7a', 'viz-context': '#eb6834', 'viz-weights': '#2a78d6',
  },
  dark: {
    surface: '#161b22', ink: '#ededed', 'ink-2': '#c8c8c8', muted: '#949494', line: '#2b2f36',
    'line-strong': '#474d57', grid: 'rgba(255,255,255,0.035)',
    harness: '#f0a04b', 'harness-soft': 'rgba(240,160,75,0.14)', predictor: '#6ea2ec',
    'predictor-soft': 'rgba(110,162,236,0.15)', optim: '#ec7b73', 'optim-soft': 'rgba(236,123,115,0.14)',
    guard: '#b08be6', 'guard-soft': 'rgba(176,139,230,0.15)', deploy: '#5cc489',
    'deploy-soft': 'rgba(92,196,137,0.14)', aux: '#858585', 'aux-soft': 'rgba(150,150,150,0.12)',
    'viz-hero': '#199e70', 'viz-context': '#d95926', 'viz-weights': '#3987e5',
  },
}

const FONTS = `
  --font-display: 'Bricolage Grotesque Variable', 'Bricolage Grotesque', 'Avenir Next', 'Segoe UI', system-ui, sans-serif;
  --font-body: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
  --font-mono: 'IBM Plex Mono', ui-monospace, 'SF Mono', Menlo, Consolas, monospace;
  --font-math: KaTeX_Math, 'Times New Roman', Times, serif;`

const vars = (theme) =>
  `svg{${Object.entries(THEMES[theme]).map(([k, v]) => `--${k}:${v};`).join('')}${FONTS}}@media (prefers-reduced-motion:reduce){*{animation:none!important}}`

// The figure rules of src/styles.css, without transitions (a scene change is a cross-fade here).
const FIGURE_CSS = `
text{font-family:var(--font-body);fill:var(--ink)}
.t-label{font-size:13px;font-weight:500}
.t-small{font-size:11.5px;fill:var(--muted)}
.t-mono{font-family:var(--font-mono);font-size:11px;letter-spacing:.04em;fill:var(--muted)}
.t-head{font-family:var(--font-display);font-size:14px;font-weight:650}
.m{font-family:var(--font-math);font-style:italic}
.m-rm{font-family:var(--font-math);font-style:normal}
.m-cal{font-family:var(--font-math);font-style:normal}
.node{--k:var(--aux);--k-soft:var(--aux-soft)}
.node[data-kind=harness]{--k:var(--harness);--k-soft:var(--harness-soft)}
.node[data-kind=predictor]{--k:var(--predictor);--k-soft:var(--predictor-soft)}
.node[data-kind=optim]{--k:var(--optim);--k-soft:var(--optim-soft)}
.node[data-kind=guard]{--k:var(--guard);--k-soft:var(--guard-soft)}
.node[data-kind=deploy]{--k:var(--deploy);--k-soft:var(--deploy-soft)}
.node[data-kind=plain]{--k:var(--line-strong);--k-soft:var(--surface)}
.node .under{fill:var(--surface);stroke:none}
.node .face{fill:var(--k-soft);stroke:var(--k);stroke-width:1.4}
.node .band{fill:var(--k);fill-opacity:.26;stroke:none}
.node[data-kind=guard] .face{stroke-dasharray:5 4}
.node[data-active=true] .face{stroke-width:2.6}
.node[data-dim=true]{opacity:.3}
.past{opacity:.62}
.edge{fill:none;stroke:var(--aux);stroke-width:1.6}
.edge[data-kind=harness]{stroke:var(--harness)}
.edge[data-kind=predictor]{stroke:var(--predictor)}
.edge[data-kind=optim]{stroke:var(--optim)}
.edge[data-kind=guard]{stroke:var(--guard);stroke-dasharray:5 5}
.edge[data-kind=deploy]{stroke:var(--deploy)}
.edge[data-dim=true]{opacity:.22}
.edge[data-flow=true]{stroke-width:2.4;stroke-dasharray:7 7;animation:flow .9s linear infinite}
@keyframes flow{to{stroke-dashoffset:-28}}
.head-plain{fill:var(--aux)}.head-harness{fill:var(--harness)}.head-predictor{fill:var(--predictor)}
.head-optim{fill:var(--optim)}.head-guard{fill:var(--guard)}.head-deploy{fill:var(--deploy)}
.ring{fill:none;stroke:var(--deploy);stroke-width:2;transform-box:fill-box;transform-origin:center;animation:pulse-ring 1.4s ease-out infinite}
@keyframes pulse-ring{0%{opacity:.7;transform:scale(.9)}100%{opacity:0;transform:scale(1.25)}}
.fig-label{font-family:var(--font-mono);font-size:11px;font-weight:500;letter-spacing:.09em;fill:var(--muted)}
.fig-title{font-family:var(--font-display);font-size:16px;font-weight:650}
.scene{font-family:var(--font-mono);font-size:11.5px;font-weight:500;letter-spacing:.04em;fill:var(--ink-2)}
.legend{font-size:13px;fill:var(--ink-2)}
.card-head{font-family:var(--font-display);font-size:15.5px;font-weight:650}
.card-tag{font-family:var(--font-mono);font-size:11.5px;fill:var(--muted)}
.card-foot{font-size:13px;fill:var(--ink-2)}
`

const KINDS = ['plain', 'harness', 'predictor', 'optim', 'guard', 'deploy']
const MARKERS = `<defs>${KINDS.map((k) =>
  `<marker id="arr-${k}" viewBox="0 0 10 10" refX="8.6" refY="5" markerWidth="9" markerHeight="9" markerUnits="userSpaceOnUse" orient="auto-start-reverse"><path d="M0.6 1 L9 5 L0.6 9 Z" class="head-${k}"/></marker>`).join('')}</defs>`

const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
const round = (n) => Math.round(n * 1000) / 1000

/* ------------------------------------------------------- render the site */

const mocks = {
  name: 'readme-figure-mocks',
  enforce: 'pre',
  resolveId(source) {
    if (/\/lib\/hooks(\.ts)?$/.test(source)) return '\0readme:hooks'
    if (/\/components\/Figure(\.tsx)?$/.test(source)) return '\0readme:figure'
  },
  load(id) {
    // A fixed scene index replaces the timer, so each render shows one scene of the loop.
    if (id === '\0readme:hooks') return `
      export const useReducedMotion = () => false
      export const useInView = () => false
      export const useFrameLoop = () => {}
      export function useCarousel(count, intervalMs) {
        globalThis.__README_LOOP__ = { count, intervalMs }
        return { step: (globalThis.__README_STEP__ ?? 0) % count, advance() {}, goto() {} }
      }`
    if (id === '\0readme:figure') return `
      export function Figure(props) {
        globalThis.__README_FIGURE__ = props
        return props.children
      }`
  },
}

async function renderScenes(server, file, name) {
  const Component = (await server.ssrLoadModule(file))[name]
  globalThis.__README_STEP__ = 0
  renderToStaticMarkup(createElement(Component))
  const { count, intervalMs } = globalThis.__README_LOOP__
  const scenes = []
  for (let step = 0; step < count; step++) {
    globalThis.__README_STEP__ = step
    const html = renderToStaticMarkup(createElement(Component))
    scenes.push({ html: tidyMath(html), figure: globalThis.__README_FIGURE__ })
  }
  return { scenes, seconds: intervalMs / 1000 }
}

const CAL = { R: 'ℛ', G: '𝒢', D: '𝒟', V: '𝒱' }

/** KaTeX fonts are not available inside an <img>, so bars become combining macrons and script letters become Unicode. */
function tidyMath(html) {
  return html
    .replace(/<tspan class="m m-bar"([^>]*)>([^<]*)<\/tspan>/g, '<tspan class="m"$1>$2̄</tspan>')
    .replace(/<tspan class="m-cal"([^>]*)>([^<]*)<\/tspan>/g, (_, a, s) => `<tspan class="m-cal"${a}>${[...s].map((c) => CAL[c] ?? c).join('')}</tspan>`)
    .replace(/​/g, '')
}

/* --------------------------------------------------------- scene compiler */

const TAG = /<(\/?)([a-zA-Z][\w:-]*)((?:[^>"']|"[^"]*"|'[^']*')*?)(\/?)>/g

/** Splits markup into its top-level elements. Groups without attributes are flattened. */
function elements(markup) {
  const out = []
  let depth = 0
  let start = 0
  const tag = new RegExp(TAG)
  for (let m; (m = tag.exec(markup)); ) {
    const [, close, , , self] = m
    if (!close && depth === 0) start = m.index
    if (close) depth--
    else if (!self) depth++
    if (depth === 0) {
      const el = markup.slice(start, tag.lastIndex)
      if (el.startsWith('<g>')) out.push(...elements(el.slice(3, -4)))
      else out.push(el)
    }
  }
  return out.filter((el) => !/^(?:<g transform="[^"]*">)?<[a-z]+[^>]*data-hidden="true"/.test(el))
}

const inner = (svg) => svg.replace(/^<svg[^>]*>/, '').replace(/<\/svg>$/, '')

/** Compiles per-scene element lists into one animated layer. */
class Timeline {
  /** The poster scene is shown wherever animation does not run, for example under reduced motion. */
  constructor(count, seconds, poster = 0, fade = 0.3) {
    this.count = count
    this.poster = poster
    this.period = count * seconds
    this.fade = (fade / this.period) * 100
    this.keyframes = new Map()
  }

  compile(frames) {
    // Merge the scenes into one paint order that respects the order inside every scene.
    const order = []
    const seen = new Map()
    frames.forEach((items, f) => {
      let last = -1
      for (const item of items) {
        let at = order.indexOf(item, last + 1)
        if (at === -1) {
          at = last + 1
          order.splice(at, 0, item)
        }
        last = at
        if (!seen.has(item)) seen.set(item, new Set())
        seen.get(item).add(f)
      }
    })
    let out = ''
    let run = null
    const flush = () => {
      if (!run) return
      out += run.cls ? `<g class="${run.cls}">${run.body}</g>` : run.body
      run = null
    }
    for (const item of order) {
      const cls = this.visibility([...seen.get(item)].sort((a, b) => a - b))
      if (run && run.cls === cls) run.body += item
      else {
        flush()
        run = { cls, body: item }
      }
    }
    flush()
    return out
  }

  visibility(frames) {
    if (frames.length === this.count) return ''
    const key = frames.join(',')
    if (!this.keyframes.has(key)) this.keyframes.set(key, `v${this.keyframes.size}`)
    return this.keyframes.get(key)
  }

  css() {
    const n = this.count
    const fp = this.fade
    let css = ''
    for (const [key, cls] of this.keyframes) {
      const set = new Set(key.split(',').map(Number))
      const stops = []
      for (let f = 0; f < n; f++) {
        if (!set.has(f) || set.has(f - 1)) continue
        let end = f
        while (end + 1 < n && set.has(end + 1)) end++
        const a = (f / n) * 100
        const b = ((end + 1) / n) * 100
        // Scene 0 is fully shown at t = 0; later scenes cross-fade in.
        if (f === 0) stops.push([0, 1])
        else stops.push([a, 0], [a + fp, 1])
        if (end === n - 1) stops.push([100, 1])
        else stops.push([b, 1], [b + fp, 0])
      }
      if (stops[0][0] > 0) stops.unshift([0, 0])
      if (stops.at(-1)[0] < 100) stops.push([100, 0])
      css += `.${cls}{opacity:${set.has(this.poster) ? 1 : 0};animation:${cls} ${round(this.period)}s linear infinite}`
      css += `@keyframes ${cls}{${stops.map(([p, o]) => `${round(p)}%{opacity:${o}}`).join('')}}`
    }
    return `${css}@media (prefers-reduced-motion:reduce){*{animation:none!important}}`
  }
}

/* ------------------------------------------------------------ figure card */

const textWidth = (s, size, mono = false) => [...s].length * size * (mono ? 0.62 : 0.56)

function grid(id, x, y, w, h) {
  return `<defs><pattern id="${id}" width="22" height="22" patternUnits="userSpaceOnUse" x="${x}" y="${y}"><path d="M22 0H0V22" fill="none" stroke="var(--grid)" stroke-width="1"/></pattern></defs><rect x="${x}" y="${y}" width="${w}" height="${h}" fill="url(#${id})"/>`
}

function legendRow(items, x, y) {
  let at = x
  return items.map(([label, color, dashed]) => {
    const mark = dashed
      ? `<line x1="${at}" x2="${at + 14}" y1="${y - 4}" y2="${y - 4}" stroke="var(--${color})" stroke-width="2" stroke-dasharray="4 3"/>`
      : `<rect x="${at}" y="${y - 8}" width="14" height="8" rx="3" fill="var(--${color})"/>`
    const s = `${mark}<text class="legend" x="${at + 21}" y="${y}">${esc(label)}</text>`
    at += 21 + textWidth(label, 13) + 26
    return s
  }).join('')
}

/** Header progress (scene label and dots), one element list per scene. */
function progressFrames(scenes, right, y) {
  return scenes.map(({ figure }) => {
    const p = figure.progress
    if (!p) return []
    let x = right
    const dots = []
    for (let i = p.count - 1; i >= 0; i--) {
      const w = i === p.index ? 18 : 6
      x -= w
      dots.unshift(`<rect x="${round(x)}" y="${y - 3}" width="${w}" height="6" rx="3" fill="var(--${i === p.index ? 'ink' : 'line-strong'})"/>`)
      x -= 4
    }
    return [`<text class="scene" x="${round(x - 8)}" y="${y + 4}" text-anchor="end">${esc(p.label)}</text>`, dots.join('')]
  })
}

function card({ theme, label, title, width, height, stage, scenes, timeline, legend, desc }) {
  const pad = 22
  const head = 52
  const foot = legend ? 44 : 0
  const W = width + pad * 2
  const H = head + height + pad * 2 + foot
  const progress = timeline.compile(progressFrames(scenes, W - 20, head / 2))
  const body = `<g transform="translate(${pad} ${head + pad})">${stage}</g>`
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="${esc(`${label}. ${title}. ${desc}`)}">
<title>${esc(`${label} · ${title}`)}</title>
<style>${vars(theme)}${FIGURE_CSS}${timeline.css()}</style>
${MARKERS}
<clipPath id="card"><rect x="0.5" y="0.5" width="${W - 1}" height="${H - 1}" rx="18"/></clipPath>
<g clip-path="url(#card)">
<rect width="${W}" height="${H}" fill="var(--surface)"/>
${grid('grid', 0, head, W, height + pad * 2)}
<line x1="0" x2="${W}" y1="${head}" y2="${head}" stroke="var(--line)"/>
${legend ? `<line x1="0" x2="${W}" y1="${H - foot}" y2="${H - foot}" stroke="var(--line)"/>` : ''}
</g>
<rect x="0.5" y="0.5" width="${W - 1}" height="${H - 1}" rx="18" fill="none" stroke="var(--line)"/>
<text class="fig-label" x="20" y="${head / 2 + 4}">${esc(label.toUpperCase())}</text>
<text class="fig-title" x="${round(20 + textWidth(label, 11, true) + 22)}" y="${head / 2 + 5.5}">${esc(title)}</text>
${progress}
${body}
${legend ? legendRow(legend, 20, H - foot / 2 + 4) : ''}
</svg>
`
}

/* ---------------------------------------------------------------- figures */

const LEGEND_STREAM = [
  ['Harness', 'harness'], ['Weights', 'predictor'], ['Distillation', 'optim'],
  ['Drift anchors', 'guard', true], ['Deployment', 'deploy'],
]

function singleSvgFigure(theme, { scenes, seconds }, { poster = 0, ...options }) {
  const viewBox = /viewBox="0 0 (\d+) (\d+)"/.exec(scenes[0].html)
  const [width, height] = [Number(viewBox[1]), Number(viewBox[2])]
  const timeline = new Timeline(scenes.length, seconds, poster)
  const stage = timeline.compile(scenes.map(({ html }) => elements(inner(/<svg[\s\S]*<\/svg>/.exec(html)[0]))))
  const { label, title } = { ...scenes[0].figure, ...options }
  return card({ theme, width, height, stage, scenes, timeline, ...options, label, title })
}

const CARD_W = 360
const CARD_GAP = 16
const CARD_HEAD = 42
const CARD_SVG = 250
const CARD_FOOT = 60

function wrap(text, max) {
  const lines = ['']
  for (const word of text.split(' ')) {
    const line = lines.at(-1)
    if (line && (line + ' ' + word).length > max) lines.push(word)
    else lines[lines.length - 1] = line ? `${line} ${word}` : word
  }
  return lines
}

/** Figure 2 renders three HTML cards; they are laid out here as one SVG. */
function paradigmsFigure(theme, { scenes, seconds }) {
  const heads = [['(a)', 'Non-parametric CL'], ['(b)', 'HERD'], ['(c)', 'Parametric CL']]
  const width = CARD_W * 3 + CARD_GAP * 2
  const height = CARD_HEAD + CARD_SVG + CARD_FOOT
  const timeline = new Timeline(scenes.length, seconds)
  let frame = ''
  heads.forEach(([tag, name], i) => {
    const x = i * (CARD_W + CARD_GAP)
    const featured = i === 1
    frame += `<rect x="${x + 0.5}" y="0.5" width="${CARD_W - 1}" height="${height - 1}" rx="14" fill="var(--surface)" stroke="var(--${featured ? 'ink' : 'line'})" stroke-width="${featured ? 2 : 1}"/>`
    frame += `<line x1="${x}" x2="${x + CARD_W}" y1="${CARD_HEAD}" y2="${CARD_HEAD}" stroke="var(--line)"/>`
    frame += `<line x1="${x}" x2="${x + CARD_W}" y1="${CARD_HEAD + CARD_SVG}" y2="${CARD_HEAD + CARD_SVG}" stroke="var(--line)"/>`
    frame += `<text x="${x + 14}" y="27"><tspan class="card-tag">${tag}</tspan><tspan class="card-head" dx="10">${esc(name)}</tspan></text>`
  })
  const frames = scenes.map(({ html }) => {
    const svgs = [...html.matchAll(/<svg[^>]*>([\s\S]*?)<\/svg>/g)].map((m) => m[1])
    const feet = [...html.matchAll(/<div class="card-foot">([\s\S]*?)<\/div>/g)].map((m) => m[1])
    return svgs.flatMap((svg, i) => {
      const x = i * (CARD_W + CARD_GAP)
      const foot = wrap(feet[i], 50).map((line, j) =>
        `<text class="card-foot" x="${x + 14}" y="${CARD_HEAD + CARD_SVG + 24 + j * 18}">${line}</text>`).join('')
      return [...elements(svg).map((el) => `<g transform="translate(${x} ${CARD_HEAD})">${el}</g>`), foot]
    })
  })
  const stage = frame + timeline.compile(frames)
  const { label, title } = scenes[0].figure
  return card({
    theme, label, title, width, height, stage, scenes, timeline,
    desc: 'Storage and inference in continual learning. HERD updates the harness and the weights, carries both forward, and deploys only the weights.',
  })
}

/* --------------------------------------------------------------- banner */

function bannerFigure(theme) {
  const font = readFileSync(join(root, 'node_modules/@fontsource-variable/bricolage-grotesque/files/bricolage-grotesque-latin-wght-normal.woff2')).toString('base64')
  const letters = [['H', 'arness-', 'harness'], ['E', 'volving', 'guard'], ['R', 'ecursive', 'predictor'], ['D', 'istillation', 'optim']]
  const period = 7.2
  const delay = (i) => `animation-delay:${round(-(letters.length - i) * (period / letters.length))}s`
  const W = 1200
  const H = 330
  const lede = ['A hybrid continual learning framework', 'that learns efficiently in the harness and', 'internalizes experience into the weights.']
  const css = `@font-face{font-family:'Bricolage Grotesque Variable';font-style:normal;font-weight:200 800;src:url(data:font/woff2;base64,${font}) format('woff2')}
.wm{font-family:var(--font-display);font-weight:800;font-size:200px;letter-spacing:-11px}
.ex{font-family:var(--font-display);font-weight:600;font-size:29px;letter-spacing:-.4px}
.ex .k{font-weight:800}
.lit{fill:var(--ink);animation:lit ${period}s linear infinite}
.rest{fill:var(--muted);animation:rest ${period}s linear infinite}
@keyframes lit{0%,22%{fill:var(--c)}25%,97%{fill:var(--ink)}100%{fill:var(--c)}}
@keyframes rest{0%,22%{fill:var(--ink)}25%,97%{fill:var(--muted)}100%{fill:var(--ink)}}
.eyebrow{font-family:var(--font-mono);font-size:13px;font-weight:500;letter-spacing:1.2px;fill:var(--muted)}
.lede{font-size:19px;fill:var(--ink-2)}
.by{font-size:14.5px;fill:var(--muted)}
.title{font-size:17px;font-weight:500;fill:var(--ink-2)}`
  const wm = letters.map(([l, , c], i) => `<tspan class="lit" style="--c:var(--${c});${delay(i)}">${l}</tspan>`).join('')
  const ex = letters.map(([l, rest, c], i) =>
    `<tspan class="k lit" style="--c:var(--${c});${delay(i)}"${i ? ' dx="12"' : ''}>${l}</tspan><tspan class="rest" style="${delay(i)}">${rest}</tspan>`).join('')
  const pills = (x, y) => `<rect x="${x}" y="${y}" width="22" height="7" rx="3.5" fill="var(--harness)"/><rect x="${x}" y="${y + 10}" width="22" height="7" rx="3.5" fill="var(--predictor)"/>`
  const stages = Array.from({ length: 8 }, (_, i) => {
    const x = 690 + i * 60
    const d = `animation-delay:${round(i * 0.45)}s`
    return `<rect class="cap" x="${x}" y="248" width="44" height="18" rx="9" fill="var(--harness-soft)" stroke="var(--harness)" style="${d}"/><rect class="cap" x="${x}" y="274" width="44" height="18" rx="9" fill="var(--predictor-soft)" stroke="var(--predictor)" style="${d}"/>`
  }).join('')
  const stageCss = `.cap{stroke-width:1.3;opacity:.25;animation:cap 3.6s ease-in-out infinite}
@keyframes cap{0%{opacity:.25}12%{opacity:1}55%{opacity:1}75%,100%{opacity:.25}}`
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="HERD: Harness-Evolving Recursive Distillation for Continual Learning">
<title>HERD: Harness-Evolving Recursive Distillation for Continual Learning</title>
<style>${vars(theme)}${css}${stageCss}</style>
<defs>
<pattern id="g" width="44" height="44" patternUnits="userSpaceOnUse"><path d="M44 0H0V44" fill="none" stroke="var(--grid)" stroke-width="1.2"/></pattern>
<radialGradient id="fade" cx="50%" cy="0%" r="90%"><stop offset="30%" stop-color="#fff"/><stop offset="80%" stop-color="#fff" stop-opacity="0"/></radialGradient>
<mask id="m"><rect width="${W}" height="${H}" fill="url(#fade)"/></mask>
</defs>
<rect width="${W}" height="${H}" fill="url(#g)" mask="url(#m)"/>
<g transform="translate(36 0)">
${pills(0, 34)}
<text class="eyebrow" x="34" y="48">CONTINUAL LEARNING OVER NLP TASK STREAMS</text>
<text class="wm" x="-8" y="214">${wm}</text>
<text class="ex" x="0" y="262">${ex}</text>
<text class="title" x="0" y="292">HERD: Harness-Evolving Recursive Distillation</text>
<text class="title" x="0" y="314">for Continual Learning</text>
</g>
<g transform="translate(690 0)">
${lede.map((line, i) => `<text class="lede" x="0" y="${118 + i * 29}">${line}</text>`).join('')}
<text class="by" x="0" y="${118 + 3 * 29 + 12}">Learn in the harness. Internalize in the weights.</text>
</g>
${stages}
<text class="t-mono" x="690" y="314" style="font-family:var(--font-mono);font-size:10.5px;letter-spacing:.6px;fill:var(--muted)">BOTH STATES CARRIED FORWARD · ONLY THE WEIGHTS ARE DEPLOYED</text>
</svg>
`
}

/** The long-horizon results, drawn by the site's own LineChart: HERD and Seq-LoRA against the other methods. */
function resultsFigure(theme, LineChart) {
  const { rows, longHorizon: lh } = JSON.parse(readFileSync(join(root, 'data/paper-results.json'), 'utf8'))
  const name = Object.fromEntries(rows.map((row) => [row.id, row.name]))
  const color = { hero: 'var(--viz-hero)', 'seq-lora': 'var(--viz-weights)' }
  const lines = (series, stages) => Object.entries(series).map(([id, values]) => ({
    id, label: name[id], color: color[id] ?? 'var(--line-strong)', strong: id === 'hero', muted: !color[id],
    points: stages.map((x, i) => ({ x, y: values[i] })),
  }))
  const chart = (title, props) => inner(renderToStaticMarkup(createElement(LineChart, props)))
    .replace(/^/, `<text class="chart-title" x="56" y="-10">${esc(title)}</text>`)
  const tick = (v) => v.toFixed(v === 0 ? 0 : 1)
  const left = chart('Mean score on the first eight tasks', {
    lines: lines(lh.firstEight.methods, lh.firstEight.stages), refs: [{ y: lh.firstEight.base, label: 'Base' }],
    x: { min: 8, max: 25, ticks: lh.firstEight.stages, label: 'Stage' }, y: { min: 0, max: 0.8, ticks: [0, 0.2, 0.4, 0.6, 0.8], format: tick },
    label: 'Mean score on tasks 1 to 8 by stage',
  })
  const right = chart('G-test macro mean', {
    lines: lines(lh.gtest.methods, lh.gtest.stages), refs: [{ y: lh.gtest.base, label: 'Base' }],
    x: { min: 0, max: 25, ticks: lh.gtest.stages, label: 'Stage' }, y: { min: 0, max: 0.9, ticks: [0, 0.3, 0.6, 0.9], format: tick },
    label: 'G-test macro mean by stage',
  })
  const stage = `<style>${CHART_CSS}</style><g transform="translate(0 24)">${left}</g><g transform="translate(460 24)">${right}</g>`
  const still = { compile: () => '', css: () => '' }
  return card({
    theme, label: 'Results', title: 'Long-horizon stability on TRACE+17', width: 900, height: 314, stage, scenes: [], timeline: still,
    legend: [['HERD', 'viz-hero'], ['Seq-LoRA', 'viz-weights'], ['Other methods', 'line-strong'], ['Base model', 'ink-2', true]],
    desc: 'HERD keeps its mean on the first eight tasks at 0.664 from stage 8 to stage 25 and loses 6.3 points of G-test, while Seq-LoRA falls to 0.372 at stage 12 and loses most of its general capability by stage 4.',
  })
}

const CHART_CSS = `.chart-axes text,.chart-ref{font-family:var(--font-mono);font-size:11.5px;fill:var(--muted)}
.chart-label{font-size:12px;font-weight:500;fill:var(--ink-2)}
.chart-dot-label{font-size:11.5px;font-weight:500;fill:var(--ink-2)}
.chart-strong{font-size:13px;font-weight:700;fill:var(--ink)}
.chart-title{font-family:var(--font-display);font-size:14px;font-weight:650;fill:var(--ink)}`

/* ------------------------------------------------------------------ main */

const server = await createServer({
  root,
  configFile: join(root, 'vite.config.ts'),
  plugins: [mocks],
  server: { middlewareMode: true, hmr: false, ws: false },
  appType: 'custom',
  logLevel: 'error',
})

try {
  const stream = await renderScenes(server, '/src/demos/StreamLoop.tsx', 'StreamLoop')
  const paradigms = await renderScenes(server, '/src/demos/Paradigms.tsx', 'Paradigms')
  const stage = await renderScenes(server, '/src/demos/StageAnatomy.tsx', 'StageAnatomy')
  const lora = await renderScenes(server, '/src/demos/LoraMerge.tsx', 'LoraMerge')
  const { LineChart } = await server.ssrLoadModule('/src/components/Charts.tsx')

  const figures = {
    banner: bannerFigure,
    'hero-stream': (theme) => singleSvgFigure(theme, stream, {
      legend: LEGEND_STREAM,
      poster: 10,
      desc: 'At each stage the harness is updated, the weights are trained by drift-constrained distillation, and the weights alone are deployed. Both states are passed to the next stage.',
    }),
    paradigms: (theme) => paradigmsFigure(theme, paradigms),
    'stage-anatomy': (theme) => singleSvgFigure(theme, stage, {
      poster: 3,
      desc: 'A bounded window of the task evolves the harness; drift-constrained distillation trains the weights under the previous-checkpoint and base-model anchors; only the weights are deployed.',
    }),
    results: (theme) => resultsFigure(theme, LineChart),
    'lora-merge': (theme) => singleSvgFigure(theme, lora, {
      poster: 2,
      label: 'Figure 4',
      desc: 'A rank-limited LoRA update is trained, merged into the inherited weights, and the merged weights are passed to the next stage and deployed.',
    }),
  }

  mkdirSync(outDir, { recursive: true })
  for (const [name, build] of Object.entries(figures)) {
    for (const theme of Object.keys(THEMES)) {
      const svg = build(theme)
      const file = join(outDir, `${name}-${theme}.svg`)
      writeFileSync(file, svg)
      console.log(`${file.slice(resolve(root, '../..').length + 1)}  ${(svg.length / 1024).toFixed(1)} KB`)
    }
  }
} finally {
  await server.close()
}
