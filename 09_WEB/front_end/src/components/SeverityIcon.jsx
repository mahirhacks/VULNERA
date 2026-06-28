const ICONS = {
  vuln: (
    <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
      <path
        fill="currentColor"
        d="M8 1.2a6.8 6.8 0 1 0 0 13.6 6.8 6.8 0 0 0 0-13.6zm2.86 4.14a.55.55 0 0 0-.78 0L8 7.42 6.92 6.34a.55.55 0 1 0-.78.78L7.22 8.2l-1.08 1.08a.55.55 0 1 0 .78.78L8 8.98l1.08 1.08a.55.55 0 0 0 .78-.78L8.78 8.2l1.08-1.08a.55.55 0 0 0 0-.78z"
      />
    </svg>
  ),
  review: (
    <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
      <path
        fill="currentColor"
        d="M8.15 1.1 15.2 13.4H1.1L8.15 1.1zm-.15 3.65a.8.8 0 0 0-.8.8v3.2a.8.8 0 0 0 1.6 0V5.55a.8.8 0 0 0-.8-.8zm0 6.1a.95.95 0 1 0 0 1.9.95.95 0 0 0 0-1.9z"
      />
    </svg>
  ),
  diffuse: (
    <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
      <path
        fill="currentColor"
        d="M8 1.2a6.8 6.8 0 1 0 0 13.6 6.8 6.8 0 0 0 0-13.6zm0 2.2a4.6 4.6 0 0 1 3.25 7.85L8 8.35 4.75 11.25A4.6 4.6 0 0 1 8 3.4zm-3.25 5.65L8 10.65l3.25-1.6A4.6 4.6 0 0 1 8 12.8a4.6 4.6 0 0 1-3.25-3.75z"
      />
    </svg>
  ),
  safe: (
    <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
      <path
        fill="currentColor"
        d="M8 1.2a6.8 6.8 0 1 0 0 13.6 6.8 6.8 0 0 0 0-13.6zm3.35 4.65a.55.55 0 0 0-.78 0L7.1 9.32 5.43 7.65a.55.55 0 1 0-.78.78l2.05 2.05a.55.55 0 0 0 .78 0l3.87-3.87a.55.55 0 0 0 0-.78z"
      />
    </svg>
  ),
};

const LABELS = {
  vuln: "Vulnerable",
  review: "Review suggested",
  diffuse: "Diffuse risk",
  safe: "Safe",
};

export default function SeverityIcon({ kind = "review", size = "md", className = "" }) {
  const resolved = ICONS[kind] ? kind : "review";

  return (
    <span
      className={`severity-icon severity-${resolved} severity-${size} ${className}`.trim()}
      title={LABELS[resolved]}
      aria-hidden="true"
    >
      {ICONS[resolved]}
    </span>
  );
}
