import { SvgDefs } from './components/Svg.tsx'
import { Abstract } from './sections/Abstract.tsx'
import { CaseStudies } from './sections/CaseStudies.tsx'
import { Cite, Footer } from './sections/Closing.tsx'
import { Experiments } from './sections/Experiments.tsx'
import { Hero } from './sections/Hero.tsx'
import { Method } from './sections/Method.tsx'
import { Nav } from './sections/Nav.tsx'

export default function App() {
  return (
    <>
      <SvgDefs />
      <Nav />
      <main>
        <Hero />
        <Abstract />
        <Method />
        <Experiments />
        <CaseStudies />
        <Cite />
      </main>
      <Footer />
    </>
  )
}
