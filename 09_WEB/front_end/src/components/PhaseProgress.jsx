import "../styles/progress.css";

export default function PhaseProgress({ job }) {
  if (!job) return null;

  const progress = Math.min(Math.max(job.progress ?? 0, 0), 1);
  const percent = Math.round(progress * 100);
  const task = job.detail || job.phase_label || "Preparing scan…";

  return (
    <div className="scan-progress">
      <div className="scan-progress-row">
        <div className="scan-progress-track" role="progressbar" aria-valuenow={percent} aria-valuemin={0} aria-valuemax={100}>
          <div className="scan-progress-fill" style={{ width: `${percent}%` }} />
        </div>
        <span className="scan-progress-pct">{percent}%</span>
      </div>
      <div className="scan-progress-task">{task}</div>
    </div>
  );
}
