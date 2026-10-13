import React from "react";
import { HERO_FULL_NAME, HERO_MODULES, type HeroModuleId } from "@hero/core";

export const MODULE_COPY: Record<HeroModuleId, { name: string; summary: string; purpose: string }> = {
  "harness-evolving": {
    name: "Harness Evolving",
    summary: "Learn from experience",
    purpose: "Use task evidence to refine the instructions and skills that guide your model in a project."
  },
  "drift-constrained-opd": {
    name: "Drift-Constrained OPD",
    summary: "Improve the model",
    purpose: "Serve the checkpoints of a stream run that internalized the harness under explicit drift limits."
  }
};

type IconName = HeroModuleId | "back" | "arrow";
const ICON_PATHS: Record<IconName, string> = {
  "harness-evolving": "M4 7h10 M18 7h2 M4 17h2 M10 17h10 M14 4v6 M6 14v6",
  "drift-constrained-opd": "M7 17H6a4 4 0 0 1-.5-8A6 6 0 0 1 17 7a5 5 0 0 1 1 10h-1 M12 12v9 M9 15l3-3 3 3",
  back: "M19 12H5 M11 6l-6 6 6 6",
  arrow: "M5 12h14 M13 6l6 6-6 6"
};

export function Icon({ name }: { name: IconName }) {
  return <svg className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={ICON_PATHS[name]} /></svg>;
}

function moduleFromLocation(): HeroModuleId {
  return HERO_MODULES.find(({ id }) => window.location.hash === `#/${id}`)?.id ?? "harness-evolving";
}

export function useModuleNavigation() {
  const [selected, setSelected] = React.useState(moduleFromLocation);
  React.useEffect(() => {
    const navigate = () => {
      if (!HERO_MODULES.some(({ id }) => window.location.hash === `#/${id}`)) return;
      setSelected(moduleFromLocation());
      window.scrollTo({ top: 0 });
      document.getElementById("workspace-content")?.focus({ preventScroll: true });
    };
    window.addEventListener("hashchange", navigate);
    return () => window.removeEventListener("hashchange", navigate);
  }, []);
  return selected;
}

type Appearance = "system" | "light" | "dark";

function AppearancePicker() {
  const [appearance, setAppearance] = React.useState<Appearance>(() => {
    const value = window.localStorage.getItem("hero.appearance");
    return value === "light" || value === "dark" ? value : "system";
  });
  React.useEffect(() => {
    document.documentElement.dataset.appearance = appearance;
    window.localStorage.setItem("hero.appearance", appearance);
  }, [appearance]);
  return <label className="appearancePicker"><span className="srOnly">Workspace appearance</span>
    <select value={appearance} onChange={(event) => setAppearance(event.target.value as Appearance)}>
      <option value="system">System theme</option><option value="light">Light theme</option><option value="dark">Dark theme</option>
    </select>
  </label>;
}

export function Workspace(props: {
  selected: HeroModuleId;
  version?: string;
  dshVersion?: string;
  harnessReady?: boolean;
  connectionFailed?: boolean;
  children: React.ReactNode;
}) {
  const copy = MODULE_COPY[props.selected];
  return <div className="appShell">
    <a className="skipLink" href="#workspace-content">Skip to workspace</a>
    <aside className="sidebar" aria-label="HERD workspace">
      <a className="hostLink" href="/"><Icon name="back" /><span>DeepSeek Harness</span></a>
      <div className="brand"><span className="brandMark" aria-hidden="true">H</span><div><strong>HERD</strong><span>Project learning</span></div></div>
      <p className="navLabel">Workspace</p>
      <nav className="moduleNav" aria-label="HERD modules">
        {HERO_MODULES.map(({ id }) => <a key={id} href={`#/${id}`} className="moduleLink" aria-current={props.selected === id ? "page" : undefined}>
          <Icon name={id} /><span><strong>{MODULE_COPY[id].name}</strong><small>{MODULE_COPY[id].summary}</small></span>
        </a>)}
      </nav>
      <div className="sidebarGuide">
        <details>
          <summary>How HERD works</summary>
          <p>Build on the experience accumulated in a project: evolve the harness from task evidence, and internalize its behavior through drift-constrained model learning.</p>
        </details>
      </div>
      <div className="sidebarFooter"><span>HERD {props.version ? `v${props.version}` : ""}</span><span>{HERO_FULL_NAME}</span></div>
    </aside>
    <div className="workspaceColumn">
      <header className="workspaceBar">
        <span className="breadcrumb">HERD <span aria-hidden="true">/</span> <strong>{copy.name}</strong></span>
        <AppearancePicker />
      </header>
      <main id="workspace-content" tabIndex={-1}>
        <header className="pageHeader">
          <div><p className="eyebrow">Project learning workspace</p><h1>{copy.name}</h1><p className="pageDescription">{copy.purpose}</p></div>
        </header>
        {props.children}
        <footer className="workspaceFooter"><span>DSH {props.dshVersion ?? (props.connectionFailed ? "unavailable" : "connecting…")}</span><span className="runtimeStatus"><span className={`dot ${props.harnessReady ? "ready" : ""}`} />Harness route {props.connectionFailed ? "unknown" : props.harnessReady === undefined ? "connecting" : props.harnessReady ? "ready" : "unavailable"}</span><span>Test examples are evaluation-only.</span></footer>
      </main>
    </div>
  </div>;
}
