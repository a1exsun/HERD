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
import { readExperiments } from './experiments.mjs'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const outDir = resolve(root, '../../docs/assets/readme')

/* ------------------------------------------------------------------ theme */

const THEMES = {
  light: {
    surface: '#ffffff', ink: '#12151c', 'ink-2': '#394150', muted: '#667085', line: '#dce0e8',
    'line-strong': '#bfc6d2', grid: 'rgba(18,21,28,0.045)',
    harness: '#d2741a', 'harness-soft': '#fbeedd', predictor: '#2f66b3', 'predictor-soft': '#e6eefa',
    optim: '#c4473f', 'optim-soft': '#fae8e6', guard: '#7548a8', 'guard-soft': '#f0e9f8',
    deploy: '#2a8552', 'deploy-soft': '#e3f3ea', aux: '#8a93a3', 'aux-soft': '#eff1f5',
  },
  dark: {
    surface: '#161b22', ink: '#ededed', 'ink-2': '#c8c8c8', muted: '#949494', line: '#2b2f36',
    'line-strong': '#474d57', grid: 'rgba(255,255,255,0.035)',
    harness: '#f0a04b', 'harness-soft': 'rgba(240,160,75,0.14)', predictor: '#6ea2ec',
    'predictor-soft': 'rgba(110,162,236,0.15)', optim: '#ec7b73', 'optim-soft': 'rgba(236,123,115,0.14)',
    guard: '#b08be6', 'guard-soft': 'rgba(176,139,230,0.15)', deploy: '#5cc489',
    'deploy-soft': 'rgba(92,196,137,0.14)', aux: '#858585', 'aux-soft': 'rgba(150,150,150,0.12)',
  },
}

const FONTS = `
  --font-display: 'Bricolage Grotesque Variable', 'Bricolage Grotesque', 'Avenir Next', 'Segoe UI', system-ui, sans-serif;
  --font-body: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
  --font-mono: 'IBM Plex Mono', ui-monospace, 'SF Mono', Menlo, Consolas, monospace;
  --font-math: KaTeX_Math, 'Times New Roman', Times, serif;`

const vars = (theme) =>
  `svg{${Object.entries(THEMES[theme]).map(([k, v]) => `--${k}:${v};`).join('')}${FONTS}}`

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
  ['Harness', 'harness'], ['Predictor', 'predictor'], ['Distillation', 'optim'],
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
    desc: 'Storage and inference in continual learning. HERD updates the harness and the predictor, carries both forward, and deploys only the predictor.',
  })
}

/* ------------------------------------------------------------- results */

function resultsFigure(theme) {
  const data = readExperiments()
  const at25 = (m) => m.depths[25] ?? m.depths['25']
  const methods = data.methods.map((m) => ({ ...m, r: at25(m) }))
  const color = { control: 'aux', 'non-parametric': 'harness', parametric: 'predictor', ours: 'deploy' }

  // (a) task mean against mathematical reasoning after 25 tasks
  const ax = { x: 70, y: 40, w: 470, h: 300 }
  const X = (v) => ax.x + (v / 0.8) * ax.w
  const Y = (v) => ax.y + ax.h - v * ax.h
  let a = `<text class="t-head" x="0" y="0">Learn new tasks, keep general skills</text>`
  a += `<text class="t-small" x="0" y="18">After 25 tasks · task mean vs. mathematical reasoning (G-test)</text>`
  for (const v of [0, 0.25, 0.5, 0.75, 1]) {
    a += `<line x1="${ax.x}" x2="${ax.x + ax.w}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--line)"/>`
    a += `<text class="t-small" x="${ax.x - 10}" y="${Y(v) + 4}" text-anchor="end">${v}</text>`
  }
  for (const v of [0, 0.2, 0.4, 0.6, 0.8]) a += `<text class="t-small" x="${X(v)}" y="${ax.y + ax.h + 20}" text-anchor="middle">${v}</text>`
  a += `<text class="t-small" x="${ax.x + ax.w / 2}" y="${ax.y + ax.h + 40}" text-anchor="middle">Task mean across 25 tasks →</text>`
  a += `<text class="t-small" transform="translate(${ax.x - 46} ${ax.y + ax.h / 2}) rotate(-90)" text-anchor="middle">Mathematical reasoning →</text>`
  const math = (m) => m.r.general.categories['mathematical-reasoning']
  for (const m of methods.filter((m) => m.id !== 'hero')) {
    a += `<circle cx="${round(X(m.r.seenMean))}" cy="${round(Y(math(m)))}" r="6" fill="var(--${color[m.family]})" fill-opacity=".85" stroke="var(--surface)" stroke-width="1.5"><title>${esc(m.short)}: ${m.r.seenMean.toFixed(3)}, ${math(m).toFixed(3)}</title></circle>`
  }
  const label = (x, y, text, anchor = 'start') => `<text class="t-small" x="${round(x)}" y="${round(y)}" text-anchor="${anchor}" style="fill:var(--ink-2)">${esc(text)}</text>`
  const by = (id) => methods.find((m) => m.id === id)
  const parametric = methods.filter((m) => m.family === 'parametric')
  const px = Math.min(...parametric.map((m) => X(m.r.seenMean)))
  const py = Math.max(...parametric.map((m) => Y(math(m))))
  a += label(px - 12, py + 4, parametric.map((m) => m.short).join(' · '), 'end')
  a += label(X(0.43), Y(0.865) + 26, methods.filter((m) => m.family !== 'parametric' && m.id !== 'hero').map((m) => m.short).join(' · '), 'middle')
  const hero = by('hero')
  const hx = round(X(hero.r.seenMean))
  const hy = round(Y(math(hero)))
  a += `<circle class="ring" cx="${hx}" cy="${hy}" r="13"/>`
  a += `<circle cx="${hx}" cy="${hy}" r="8" fill="var(--deploy)" stroke="var(--surface)" stroke-width="2"><title>HERD: ${hero.r.seenMean.toFixed(3)}, ${math(hero).toFixed(3)}</title></circle>`
  a += `<text x="${hx + 16}" y="${hy + 5}" style="font-weight:650;fill:var(--deploy);font-size:14px">HERD</text>`

  // (b) general capability on the 200-item probe through the stream
  const bx = { x: 50, y: 40, w: 360, h: 300 }
  const BX = (s) => bx.x + (s / 25) * bx.w
  const BY = (v) => bx.y + bx.h - v * bx.h
  let b = `<text class="t-head" x="0" y="0">General capability through the stream</text>`
  b += `<text class="t-small" x="0" y="18">Category macro on the 200-item G probe · x: tasks learned</text>`
  for (const v of [0, 0.25, 0.5, 0.75, 1]) {
    b += `<line x1="${bx.x}" x2="${bx.x + bx.w}" y1="${BY(v)}" y2="${BY(v)}" stroke="var(--line)"/>`
    b += `<text class="t-small" x="${bx.x - 10}" y="${BY(v) + 4}" text-anchor="end">${v}</text>`
  }
  for (const s of [0, 4, 8, 16, 25]) b += `<text class="t-small" x="${BX(s)}" y="${bx.y + bx.h + 20}" text-anchor="middle">${s}</text>`
  const series = [['hero', 'deploy', 3], ['uniform-replay', 'predictor', 2], ['base', 'muted', 2, true]]
  series.forEach(([id, c, w, dashed], i) => {
    const m = by(id)
    const pts = m.constant
      ? [{ stage: 0, value: m.series.generalProbe[0].value }, { stage: 25, value: m.series.generalProbe[0].value }]
      : m.series.generalProbe
    const d = pts.map((p) => `${round(BX(p.stage))},${round(BY(p.value))}`).join(' ')
    b += `<polyline points="${d}" fill="none" stroke="var(--${c})" stroke-width="${w}" stroke-linejoin="round" stroke-linecap="round" ${dashed ? 'stroke-dasharray="5 5"' : `pathLength="1" class="draw d${i}"`}/>`
    const last = pts.at(-1)
    b += `<text x="${round(BX(25) + 8)}" y="${round(BY(last.value) + (id === 'uniform-replay' ? 12 : id === 'hero' ? -4 : 4))}" style="font-size:12.5px;font-weight:600;fill:var(--${c})">${esc(m.short)}</text>`
  })

  const W = 1180
  const H = 540
  const legend = [['This work', 'deploy'], ['Weights', 'predictor'], ['Context', 'harness'], ['Control', 'aux']]
  const css = `.draw{stroke-dasharray:1 1;animation:draw 7s cubic-bezier(.4,0,.2,1) infinite}.d1{animation-delay:.25s}
@keyframes draw{0%{stroke-dashoffset:1;opacity:1}55%{stroke-dashoffset:0;opacity:1}92%{stroke-dashoffset:0;opacity:1}100%{stroke-dashoffset:0;opacity:0}}`
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="Results on TRACE+17 after 25 tasks. Left: HERD reaches a task mean of ${hero.r.seenMean.toFixed(3)} while keeping mathematical reasoning at ${math(hero).toFixed(3)}; weight-based learners reach ${Math.min(...parametric.map((m) => m.r.seenMean)).toFixed(2)}–${Math.max(...parametric.map((m) => m.r.seenMean)).toFixed(2)} with mathematical reasoning at ${Math.min(...parametric.map(math)).toFixed(2)}–${Math.max(...parametric.map(math)).toFixed(2)}. Right: HERD's general capability rises from ${hero.series.generalProbe[0].value} to ${hero.series.generalProbe.at(-1).value} through the stream.">
<title>Results · TRACE+17 after 25 tasks</title>
<style>${vars(theme)}${FIGURE_CSS}${css}</style>
<rect x="0.5" y="0.5" width="${W - 1}" height="${H - 1}" rx="18" fill="var(--surface)" stroke="var(--line)"/>
<text class="fig-label" x="20" y="30">RESULTS</text>
<text class="fig-title" x="110" y="31.5">TRACE+17 · Qwen3.5-4B · 25 tasks · HERD: mean of seeds 42, 43, 44</text>
<line x1="0" x2="${W}" y1="52" y2="52" stroke="var(--line)"/>
<g transform="translate(28 92)">${a}</g>
<g transform="translate(630 92)">${b}</g>
<line x1="0" x2="${W}" y1="${H - 44}" y2="${H - 44}" stroke="var(--line)"/>
${legendRow(legend, 20, H - 18)}
</svg>
`
}

/* --------------------------------------------------------------- banner */

function bannerFigure(theme) {
  const font = readFileSync(join(root, 'node_modules/@fontsource-variable/bricolage-grotesque/files/bricolage-grotesque-latin-wght-normal.woff2')).toString('base64')
  const letters = [['H', 'arness-', 'harness'], ['E', 'volving', 'guard'], ['R', 'ecursive', 'predictor'], ['D', 'istillation', 'optim']]
  const period = 7.2
  const delay = (i) => `animation-delay:${round(-(letters.length - i) * (period / letters.length))}s`
  const W = 1200
  const H = 330
  const lede = ['A continual-learning framework that distills an', 'evolving, bounded harness into a standalone', 'predictor under two behavioral drift constraints.']
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
<text class="by" x="0" y="${118 + 3 * 29 + 12}">Paper under review · ACL Rolling Review, October 2026</text>
</g>
${stages}
<text class="t-mono" x="690" y="314" style="font-family:var(--font-mono);font-size:10.5px;letter-spacing:.6px;fill:var(--muted)">BOTH STATES CARRIED FORWARD · ONLY THE PREDICTOR IS DEPLOYED</text>
</svg>
`
}

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

  const figures = {
    banner: bannerFigure,
    'hero-stream': (theme) => singleSvgFigure(theme, stream, {
      legend: LEGEND_STREAM,
      poster: 10,
      desc: 'At each stage the harness is updated, the predictor is trained by Drift-Constrained OPD, and the predictor alone is deployed. Both states are passed to the next stage.',
    }),
    paradigms: (theme) => paradigmsFigure(theme, paradigms),
    'stage-anatomy': (theme) => singleSvgFigure(theme, stage, {
      poster: 3,
      desc: 'Bounded current evidence updates the harness checkpoint; Drift-Constrained OPD trains the predictor under history and general-capability anchors; only the predictor is deployed.',
    }),
    'lora-merge': (theme) => singleSvgFigure(theme, lora, {
      poster: 2,
      label: 'Figure 4',
      desc: 'A rank-limited LoRA update is trained, merged into the inherited weights, and the merged predictor is passed to the next stage and deployed.',
    }),
    results: resultsFigure,
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
