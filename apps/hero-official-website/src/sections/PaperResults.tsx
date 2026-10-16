import paper from '../../data/paper-results.json'

const directions = [1, 1, -1, 1, 1, 1, 1]
const learning = paper.rows.filter((row) => row.family !== 'Control')
const best = directions.map((direction, i) => Math.max(...learning.map((row) => direction * row.values[i])))

export function PaperResults() {
  const hero = paper.rows[0].values
  return (
    <>
      <div className="experiment-stats">
        {[[0, 'HERD task mean'], [1, 'HERD acquisition'], [2, 'HERD forgetting'], [4, 'HERD G-test']].map(([i, label]) => (
          <div key={label}><span>{label}</span><strong>{hero[Number(i)].toFixed(3)}</strong><small>Paper · Table 1</small></div>
        ))}
      </div>
      <div className="experiment-scroll" role="region" aria-label="Paper Table 1" tabIndex={0}>
        <table className="experiment-table">
          <caption>Paper Table 1 · after 25 tasks · means over three seeds, rounded as printed</caption>
          <thead><tr><th scope="col">Method</th>{paper.columns.map((name, i) => <th scope="col" key={name}>{name} {directions[i] > 0 ? '↑' : '↓'}</th>)}</tr></thead>
          <tbody>{paper.rows.map((row) => (
            <tr key={row.id} className={row.id === 'hero' ? 'experiment-hero' : undefined}>
              <th scope="row">{row.name}<small>{row.family}</small></th>
              {row.values.map((value, i) => <td key={paper.columns[i]} className={row.family !== 'Control' && value * directions[i] === best[i] ? 'experiment-best' : undefined}>{value.toFixed(3)}</td>)}
            </tr>
          ))}</tbody>
        </table>
      </div>
      <p className="experiment-reading">
        HERD reaches 0.706 final task mean with acquisition of 0.713, close to Seq-LoRA’s 0.715.
        Forgetting is 0.010 and general capability drops by 0.063. SDFT’s lower forgetting accompanies
        collapsed acquisition (0.024), rather than successful retention.
      </p>
    </>
  )
}
