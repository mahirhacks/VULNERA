const KEYWORDS = new Set([
  "alignas", "alignof", "asm", "auto", "bool", "break", "case", "catch", "char", "class",
  "const", "constexpr", "continue", "default", "delete", "do", "double", "else", "enum",
  "extern", "false", "float", "for", "goto", "if", "inline", "int", "long", "namespace",
  "new", "nullptr", "private", "protected", "public", "register", "return", "short",
  "signed", "sizeof", "static", "struct", "switch", "template", "this", "throw", "true",
  "try", "typedef", "typename", "union", "unsigned", "using", "virtual", "void",
  "volatile", "while",
]);

function escapeHtml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function highlightPlain(segment) {
  return escapeHtml(segment).replace(/\b([A-Za-z_]\w*)\b/g, (match) =>
    KEYWORDS.has(match) ? `<span class="tok-keyword">${match}</span>` : match,
  );
}

export function highlightLine(line) {
  if (!line) return " ";

  const stripped = line.trimStart();
  if (stripped.startsWith("#")) {
    const leading = line.slice(0, line.length - stripped.length);
    return `${escapeHtml(leading)}<span class="tok-preprocessor">${escapeHtml(stripped)}</span>`;
  }

  const commentIndex = line.indexOf("//");
  const codePart = commentIndex === -1 ? line : line.slice(0, commentIndex);
  const commentPart = commentIndex === -1 ? "" : line.slice(commentIndex);

  const stringRe = /("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')/g;
  let html = "";
  let last = 0;
  let match;
  while ((match = stringRe.exec(codePart)) !== null) {
    html += highlightPlain(codePart.slice(last, match.index));
    html += `<span class="tok-string">${escapeHtml(match[0])}</span>`;
    last = match.index + match[0].length;
  }
  html += highlightPlain(codePart.slice(last));
  if (commentPart) {
    html += `<span class="tok-comment">${escapeHtml(commentPart)}</span>`;
  }
  return html || " ";
}
