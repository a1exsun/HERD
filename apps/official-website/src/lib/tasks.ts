export interface StreamTask {
  name: string
  kind: string
}

/** TRACE-8 in its canonical order, the primary stream of the experimental protocol. */
export const TRACE8: StreamTask[] = [
  { name: 'C-STANCE', kind: 'stance detection' },
  { name: 'FOMC', kind: 'policy classification' },
  { name: 'MeetingBank', kind: 'meeting summarization' },
  { name: 'Py150', kind: 'code completion' },
  { name: 'ScienceQA', kind: 'science QA' },
  { name: 'NumGLUE-cm', kind: 'numerical reasoning' },
  { name: 'NumGLUE-ds', kind: 'numerical reasoning' },
  { name: '20Minuten', kind: 'text simplification' },
]

/** One hue per task, used only to tell retained examples apart by origin. */
export const TASK_HUES = ['#d2741a', '#2f66b3', '#2a8552', '#7548a8', '#c4473f', '#0f8a8a', '#a0628f', '#7a7f2a']
