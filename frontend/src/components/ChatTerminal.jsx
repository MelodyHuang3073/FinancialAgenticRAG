import React, { useState, useRef, useEffect } from 'react';
import {
  Plus, Send, Mic, FileText, Bot, User,
  Code2, ShieldCheck, ChevronDown, ChevronUp,
  X, Sparkles, TrendingUp, BookOpen, Globe,
  Paperclip, AlertCircle, CheckCircle
} from 'lucide-react';

/* ─── Markdown Table renderer ─── */
function MarkdownTable({ text }) {
  const lines = text.split('\n').map(l => l.trim()).filter(Boolean);
  // Detect if this looks like a pipe-delimited table block
  const tableLines = lines.filter(l => l.startsWith('|') || l.includes(' | '));
  if (tableLines.length < 2) return <pre style={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: 12, color: '#334155', margin: 0 }}>{text}</pre>;

  // Parse header + separator + rows
  const rows = lines.map(l => l.split('|').map(c => c.trim()).filter(c => c !== ''));
  const isSeparator = (row) => row.every(c => /^[-:]+$/.test(c));
  const headerRowIdx = rows.findIndex((r, i) => i + 1 < rows.length && isSeparator(rows[i + 1]));

  if (headerRowIdx === -1) {
    // No proper header found — render as simple definition rows
    return (
      <table style={{ borderCollapse: 'collapse', width: '100%', fontSize: 12 }}>
        <tbody>
          {rows.map((row, ri) => (
            <tr key={ri} style={{ background: ri % 2 === 0 ? '#f8fafc' : '#fff' }}>
              {row.map((cell, ci) => (
                <td key={ci} style={{ border: '1px solid #e2e8f0', padding: '4px 8px', color: '#334155' }}>{cell}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    );
  }

  const headerRow = rows[headerRowIdx];
  const dataRows = rows.filter((_, i) => i !== headerRowIdx && !isSeparator(rows[i]));

  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ borderCollapse: 'collapse', width: '100%', fontSize: 12 }}>
        <thead>
          <tr style={{ background: '#f1f5f9' }}>
            {headerRow.map((h, i) => (
              <th key={i} style={{ border: '1px solid #cbd5e1', padding: '5px 10px', textAlign: 'left', fontWeight: 700, color: '#1e293b', whiteSpace: 'nowrap' }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {dataRows.map((row, ri) => (
            <tr key={ri} style={{ background: ri % 2 === 0 ? '#fff' : '#f8fafc' }}>
              {row.map((cell, ci) => (
                <td key={ci} style={{ border: '1px solid #e2e8f0', padding: '4px 10px', color: ci === 0 ? '#0f172a' : '#334155', fontWeight: ci === 0 ? 600 : 400 }}>{cell}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ─── Markdown-like bold renderer ─── */
function AnswerText({ text }) {
  if (!text) return null;
  const lines = text.split('\n');
  return (
    <div style={{ lineHeight: 1.75, fontSize: 15, color: '#0d0d0d' }}>
      {lines.map((line, i) => {
        if (line.startsWith('### ')) return (
          <h3 key={i} style={{ fontWeight: 700, fontSize: 16, margin: '16px 0 6px', color: '#0d0d0d' }}>
            {line.replace('### ', '')}
          </h3>
        );
        if (line.startsWith('**') && line.endsWith('**')) return (
          <p key={i} style={{ fontWeight: 600, margin: '4px 0' }}>{line.slice(2, -2)}</p>
        );
        if (line.startsWith('- ')) return (
          <div key={i} style={{ display: 'flex', gap: 8, margin: '3px 0', paddingLeft: 8 }}>
            <span style={{ color: '#888', flexShrink: 0 }}>•</span>
            <span>{line.slice(2)}</span>
          </div>
        );
        if (line.startsWith('> ')) return (
          <blockquote key={i} style={{
            borderLeft: '3px solid #d9d9e3', paddingLeft: 14, margin: '6px 0',
            color: '#555', fontStyle: 'italic', fontSize: 14,
          }}>
            {line.slice(2)}
          </blockquote>
        );
        if (line.trim() === '') return <div key={i} style={{ height: 8 }} />;
        return <p key={i} style={{ margin: '4px 0' }}>{line}</p>;
      })}
    </div>
  );
}

function splitAnswerText(text) {
  if (!text) return { summaryText: '', detailText: '' };

  const lines = text.split('\n');
  const firstHeadingIndex = lines.findIndex((line) => line.startsWith('### '));

  if (firstHeadingIndex !== -1) {
    return {
      summaryText: lines.slice(0, firstHeadingIndex).join('\n').trim(),
      detailText: lines.slice(firstHeadingIndex).join('\n').trim(),
    };
  }

  const nonEmptyLines = lines.filter((line) => line.trim());
  if (nonEmptyLines.length <= 3) {
    return { summaryText: text.trim(), detailText: '' };
  }

  const summaryLines = [];
  for (const line of nonEmptyLines) {
    if (summaryLines.length >= 3) break;
    summaryLines.push(line.trim());
  }

  return {
    summaryText: summaryLines.join('\n').trim(),
    detailText: nonEmptyLines.slice(summaryLines.length).join('\n').trim(),
  };
}

function formatResultValue(value) {
  if (value === null || value === undefined || value === '') return '無可顯示結果';
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value.toLocaleString('zh-TW', { maximumFractionDigits: 6 });
  }
  return String(value);
}

function ResultSummaryCard({ value, label = '最終計算結果' }) {
  if (value === null || value === undefined || value === '') return null;

  return (
    <div style={{
      marginBottom: 16,
      padding: '14px 16px',
      borderRadius: 14,
      background: 'linear-gradient(135deg, #f0fdf4 0%, #ecfeff 100%)',
      border: '1px solid #bbf7d0',
      boxShadow: '0 1px 3px rgba(0,0,0,0.04)',
    }}>
      <div style={{ fontSize: 11, fontWeight: 700, color: '#166534', letterSpacing: '0.04em', textTransform: 'uppercase' }}>
        {label}
      </div>
      <div style={{ marginTop: 6, fontSize: 28, fontWeight: 800, color: '#065f46', lineHeight: 1.1 }}>
        {formatResultValue(value)}
      </div>
      <div style={{ marginTop: 6, fontSize: 12, color: '#047857' }}>
        Review the number first, then expand the reasoning below.
      </div>
    </div>
  );
}

/* ─── Code Block ─── */
function CodeBlock({ code, log }) {
  const [copied, setCopied] = useState(false);
  return (
    <div style={{ borderRadius: 10, overflow: 'hidden', border: '1px solid #2a2a2a', marginTop: 10 }}>
      {/* Header bar */}
      <div style={{ background: '#1e1e1e', padding: '8px 14px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ color: '#aaa', fontSize: 11, fontFamily: 'monospace', fontWeight: 500 }}>python · PoT Sandbox</span>
        <button onClick={() => { navigator.clipboard.writeText(code); setCopied(true); setTimeout(() => setCopied(false), 2000); }}
          style={{ color: '#aaa', fontSize: 11, background: 'none', border: 'none', cursor: 'pointer' }}>
          {copied ? '✓ Copied' : 'Copy code'}
        </button>
      </div>
      {/* Code */}
      <pre style={{
        background: '#0d0d0d', color: '#4ec9b0', padding: '14px 16px',
        fontSize: 12, lineHeight: 1.65, overflowX: 'auto',
        fontFamily: '"JetBrains Mono", "Fira Code", monospace', margin: 0,
      }}>{code}</pre>
      {log && (
        <div style={{ background: '#1a1a1a', padding: '8px 14px', borderTop: '1px solid #2a2a2a' }}>
          <span style={{ color: '#fbbf24', fontSize: 11, fontFamily: 'monospace', fontWeight: 600 }}>⚡ Output: </span>
          <span style={{ color: '#86efac', fontSize: 11, fontFamily: 'monospace' }}>{log}</span>
        </div>
      )}
    </div>
  );
}

/* ─── Tri-Check Badge ─── */
function TriCheckBadge({ verification }) {
  if (!verification) return null;
  const checks = verification.checks || {};
  const accepted = verification.decision === 'ACCEPT';
  const conf = Math.round((verification.confidence_score || 0) * 100);

  return (
    <div style={{
      display: 'flex', flexWrap: 'wrap', gap: 8,
      marginTop: 12, paddingTop: 12, borderTop: '1px solid #e5e5e5',
    }}>
      {[
        { key: 'nu_suff', label: 'ν_suff 資料充分性' },
        { key: 'nu_num',  label: 'ν_num 算術一致性' },
        { key: 'nu_cross',label: 'ν_cross 跨期驗證' },
      ].map(({ key, label }) => {
        const passed = checks[key]?.passed;
        return (
          <span key={key} style={{
            display: 'inline-flex', alignItems: 'center', gap: 5,
            fontSize: 11, fontFamily: 'monospace', fontWeight: 600,
            padding: '3px 10px', borderRadius: 999,
            background: passed ? '#f0fdf4' : '#fff1f2',
            color: passed ? '#15803d' : '#b91c1c',
            border: `1px solid ${passed ? '#bbf7d0' : '#fecdd3'}`,
          }}>
            {passed ? <CheckCircle size={10} /> : <AlertCircle size={10} />}
            {label}
          </span>
        );
      })}
      <span style={{
        fontSize: 11, fontFamily: 'monospace', padding: '3px 10px', borderRadius: 999,
        background: accepted ? '#eff6ff' : '#fff7ed',
        color: accepted ? '#1d4ed8' : '#c2410c',
        border: `1px solid ${accepted ? '#bfdbfe' : '#fed7aa'}`,
        fontWeight: 700,
      }}>
        {accepted ? `✅ ACCEPT · ${conf}%` : `⚠️ REJECT · ${conf}%`}
      </span>
    </div>
  );
}

/* ─── Single Message Bubble ─── */
function MessagePair({ msg, idx, openTrace, setOpenTrace }) {
  const isOpen = openTrace === idx;
  const [showDetails, setShowDetails] = useState(true);
  const [showReasoning, setShowReasoning] = useState(true);
  const [showTraceDetails, setShowTraceDetails] = useState(false);
  const [expandedEvidence, setExpandedEvidence] = useState(null);
  const { summaryText, detailText } = splitAnswerText(msg.result.final_answer);

  return (
    <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 28 }}>
      <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
        <div style={{
          maxWidth: 520, background: '#f4f4f4', color: '#0d0d0d',
          padding: '12px 18px', borderRadius: 20, fontSize: 15,
          border: '1px solid #e5e5e5', lineHeight: 1.6,
        }}>
          {msg.query}
        </div>
      </div>

      <div style={{ display: 'flex', gap: 14, alignItems: 'flex-start' }}>
        <div style={{
          width: 32, height: 32, borderRadius: '50%', background: '#10a37f',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          flexShrink: 0, marginTop: 2,
        }}>
          <Bot size={16} color="#fff" />
        </div>

        <div style={{
          flex: 1, background: '#fff', border: '1px solid #e5e5e5',
          borderRadius: 18, padding: '20px 24px',
          boxShadow: '0 1px 6px rgba(0,0,0,0.04)',
        }}>
          <ResultSummaryCard value={msg.result.result_value} />

          {summaryText && (
            <div style={{ marginBottom: 12 }}>
              <AnswerText text={summaryText} />
            </div>
          )}

          {(detailText || msg.result.evidence_sources?.length > 0) && (
            <div style={{ marginTop: 8, paddingTop: 12, borderTop: '1px solid #f0f0f0' }}>
              <button
                onClick={() => setShowDetails((prev) => !prev)}
                style={{
                  display: 'flex', alignItems: 'center', gap: 6,
                  fontSize: 12, fontFamily: 'monospace', color: '#6366f1',
                  background: 'none', border: 'none', cursor: 'pointer', padding: 0,
                }}
              >
                <ShieldCheck size={13} />
                {showDetails ? 'Collapse' : 'Show'} detailed logic
                {showDetails ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
              </button>

              {showDetails && (
                <>
                  {detailText && (
                    <div style={{ marginTop: 12 }}>
                      <AnswerText text={detailText} />
                    </div>
                  )}

                  {msg.result.evidence_sources?.length > 0 && (
                    <div style={{ marginTop: 16, paddingTop: 14, borderTop: '1px solid #f0f0f0' }}>
                      <p style={{ fontSize: 11, color: '#888', fontWeight: 600, marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                        Source Evidence ({msg.result.evidence_sources.length} chunks)
                      </p>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                        {msg.result.evidence_sources.map((ev, si) => {
                          const isExpanded = expandedEvidence === si;
                          const fullContent = ev.content || '';
                          const isTable = fullContent.includes(' | ') && fullContent.split('\n').some(l => l.includes('|'));

                          // Section colour coding
                          const sectionColors = {
                            income_statement: { bg: '#eff6ff', text: '#1d4ed8', border: '#bfdbfe' },
                            balance_sheet:    { bg: '#f0fdf4', text: '#15803d', border: '#bbf7d0' },
                            cash_flow:        { bg: '#fefce8', text: '#a16207', border: '#fde68a' },
                            notes_general:    { bg: '#fdf4ff', text: '#7e22ce', border: '#e9d5ff' },
                            mda:              { bg: '#fff7ed', text: '#c2410c', border: '#fed7aa' },
                          };
                          const sc = sectionColors[ev.section] || { bg: '#f8fafc', text: '#64748b', border: '#e2e8f0' };

                          return (
                            <div key={si} style={{ border: `1px solid ${sc.border}`, borderRadius: 10, background: '#fafafa' }}>
                              <button
                                onClick={() => setExpandedEvidence(isExpanded ? null : si)}
                                style={{
                                  width: '100%', textAlign: 'left', background: 'none', border: 'none',
                                  padding: '10px 12px', cursor: 'pointer', display: 'flex', justifyContent: 'space-between', gap: 8,
                                }}
                              >
                                <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                                  {/* Title row */}
                                  <span style={{ fontSize: 12, color: '#334155', fontWeight: 600 }}>
                                    [{ev.company || 'Company'}] {ev.table_name || 'Source chunk'}
                                  </span>

                                  {/* Badge row: section + chunk_type */}
                                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5 }}>
                                    {ev.section && (
                                      <span style={{
                                        fontSize: 10, fontFamily: 'monospace', fontWeight: 700,
                                        padding: '1px 7px', borderRadius: 999,
                                        background: sc.bg, color: sc.text, border: `1px solid ${sc.border}`,
                                      }}>
                                        {ev.section}
                                      </span>
                                    )}
                                    {ev.chunk_type && (
                                      <span style={{
                                        fontSize: 10, fontFamily: 'monospace',
                                        padding: '1px 7px', borderRadius: 999,
                                        background: ev.chunk_type === 'table_row' ? '#fef9c3' : '#f1f5f9',
                                        color: ev.chunk_type === 'table_row' ? '#854d0e' : '#475569',
                                        border: `1px solid ${ev.chunk_type === 'table_row' ? '#fde68a' : '#e2e8f0'}`,
                                      }}>
                                        {ev.chunk_type === 'table_row' ? '📊 table_row' : '📝 ' + ev.chunk_type}
                                      </span>
                                    )}
                                    {ev.period && (
                                      <span style={{
                                        fontSize: 10, fontFamily: 'monospace',
                                        padding: '1px 7px', borderRadius: 999,
                                        background: '#f8fafc', color: '#64748b', border: '1px solid #e2e8f0',
                                      }}>
                                        {ev.period}
                                      </span>
                                    )}
                                    <span style={{
                                      fontSize: 10, fontFamily: 'monospace',
                                      padding: '1px 7px', borderRadius: 999,
                                      background: '#f8fafc', color: '#94a3b8', border: '1px solid #e2e8f0',
                                    }}>
                                      score: {(ev.relevance_score || 0).toFixed(3)}
                                    </span>
                                  </div>

                                  {/* Sub-question label */}
                                  {ev.sub_question && (
                                    <span style={{ fontSize: 10, color: '#6366f1', fontFamily: 'monospace' }}>
                                      ↳ 検索子問題: {ev.sub_question}
                                    </span>
                                  )}
                                </div>
                                <span style={{ fontSize: 11, color: '#6366f1', flexShrink: 0 }}>{isExpanded ? 'Collapse' : 'View content'}</span>
                              </button>
                              {isExpanded && (
                                <div style={{ padding: '0 12px 14px' }}>
                                  {isTable
                                    ? <MarkdownTable text={fullContent} />
                                    : <pre style={{ whiteSpace: 'pre-wrap', fontSize: 12, lineHeight: 1.7, color: '#475569', margin: 0, fontFamily: 'monospace' }}>{fullContent || 'No chunk content available'}</pre>
                                  }
                                </div>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}

                  <div style={{ marginTop: 14, paddingTop: 12, borderTop: '1px solid #f0f0f0' }}>
                    <button
                      onClick={() => setShowReasoning((prev) => !prev)}
                      style={{
                        display: 'flex', alignItems: 'center', gap: 6,
                        fontSize: 12, fontFamily: 'monospace', color: '#6366f1',
                        background: 'none', border: 'none', cursor: 'pointer', padding: 0,
                      }}
                    >
                      <BookOpen size={13} />
                      {showReasoning ? 'Collapse' : 'Show'} reasoning and routing
                      {showReasoning ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
                    </button>

                    {showReasoning && (
                      <div style={{ marginTop: 12 }}>
                        <div style={{ display: 'flex', gap: 20, marginBottom: 12, flexWrap: 'wrap' }}>
                          <span style={{ fontSize: 12, fontFamily: 'monospace', color: '#888' }}>
                            Answer mode: <b style={{ color: '#d97706' }}>{msg.result.answer_mode}</b>
                          </span>
                          <span style={{ fontSize: 12, fontFamily: 'monospace', color: '#888' }}>
                            Complexity: <b style={{ color: '#d97706' }}>{msg.result.complexity}</b>
                          </span>
                          <span style={{ fontSize: 12, fontFamily: 'monospace', color: '#888' }}>
                            Iterations: <b style={{ color: '#0284c7' }}>{msg.result.total_iterations} / 3</b>
                          </span>
                          {msg.result.question_type && (
                            <>
                              <span style={{ fontSize: 12, fontFamily: 'monospace', color: '#888' }}>
                                Question Type: <b style={{ color: '#059669' }}>{msg.result.question_type}</b>
                              </span>
                              <span style={{ fontSize: 12, fontFamily: 'monospace', color: '#888' }}>
                                Task: <b style={{ color: '#059669' }}>{msg.result.cognitive_task}</b>
                              </span>
                              <span style={{ fontSize: 12, fontFamily: 'monospace', color: '#888' }}>
                                Strategy: <b style={{ color: '#059669' }}>{msg.result.retrieval_strategy}</b>
                              </span>
                            </>
                          )}
                        </div>

                        <div style={{ padding: '10px 12px', background: '#fff', border: '1px solid #e5e7eb', borderRadius: 10 }}>
                          <div style={{ fontSize: 11, color: '#64748b', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.04em', marginBottom: 8 }}>Reasoning steps</div>
                          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                            {msg.result.reasoning_steps?.map((step, index) => {
                              const typeColors = {
                                classification: '#7c3aed',
                                decomposition:  '#0284c7',
                                step_retrieval: '#0f766e',
                                iteration:      '#b45309',
                                refinement:     '#dc2626',
                                retrieval:      '#0f766e',
                              };
                              const color = typeColors[step.type] || '#334155';
                              return (
                                <div key={index} style={{ fontSize: 12.5, color: '#334155', lineHeight: 1.5, paddingBottom: 4, borderBottom: '1px dashed #e5e7eb' }}>
                                  <span style={{ fontWeight: 700, color }}>{index + 1}. {step.step_name}</span>
                                  {step.detail && <div style={{ marginTop: 2, color: '#475569', fontSize: 12 }}>{step.detail}</div>}
                                  {/* Show sub_questions list for decomposition step */}
                                  {step.type === 'decomposition' && step.sub_questions?.length > 0 && (
                                    <div style={{ marginTop: 4, paddingLeft: 12, borderLeft: '2px solid #e0e7ff' }}>
                                      {step.sub_questions.map((sq, sqi) => (
                                        <div key={sqi} style={{ fontSize: 11.5, color: '#6366f1', marginTop: 2 }}>
                                          <b>Step {sq.step}</b> [{sq.type}]{sq.target_metric ? ` • metric: ${sq.target_metric}` : ''}{sq.target_year ? ` • year: ${sq.target_year}` : ''}<br/>
                                          <span style={{ color: '#64748b' }}>{sq.query}</span>
                                        </div>
                                      ))}
                                    </div>
                                  )}
                                  {/* Show retrieved count for step_retrieval */}
                                  {step.type === 'step_retrieval' && step.detail && (
                                    <div style={{ marginTop: 2, fontSize: 11, color: '#0f766e', fontFamily: 'monospace' }}>{step.detail}</div>
                                  )}
                                </div>
                              );
                            })}
                          </div>
                        </div>

                        <div style={{ marginTop: 10 }}>
                          <button
                            onClick={() => setShowTraceDetails((prev) => !prev)}
                            style={{
                              display: 'flex', alignItems: 'center', gap: 6,
                              fontSize: 12, fontFamily: 'monospace', color: '#6366f1',
                              background: 'none', border: 'none', cursor: 'pointer', padding: 0,
                            }}
                          >
                            <Code2 size={13} />
                            {showTraceDetails ? 'Collapse' : 'Show'} full trace
                            {showTraceDetails ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
                          </button>

                          {showTraceDetails && (msg.result.execution_trace || msg.result.reasoning_steps)?.length > 0 && (
                            <div style={{ marginTop: 10, display: 'flex', flexDirection: 'column', gap: 8 }}>
                              {(msg.result.execution_trace || msg.result.reasoning_steps).map((step, index) => (
                                <div key={index} style={{ padding: '10px 12px', borderRadius: 10, background: '#f8fafc', border: '1px solid #e5e7eb' }}>
                                  <div style={{ fontSize: 13, fontWeight: 700, color: '#0f766e' }}>{index + 1}. {step.step_name}</div>
                                  {step.detail && <div style={{ marginTop: 4, fontSize: 12, color: '#475569' }}>{step.detail}</div>}
                                  {step.data?.retrieved_passages?.length > 0 && (
                                    <div style={{ marginTop: 6, fontSize: 12, color: '#475569' }}>
                                      <div style={{ fontWeight: 600, marginBottom: 4 }}>Retrieved snippets</div>
                                      {step.data.retrieved_passages.slice(0, 2).map((passage, passageIdx) => (
                                        <div key={passageIdx} style={{ padding: '6px 0', borderTop: passageIdx ? '1px solid #e5e7eb' : 'none' }}>
                                          {passage.snippet}
                                        </div>
                                      ))}
                                    </div>
                                  )}
                                  {step.data?.verification && (
                                    <div style={{ marginTop: 6, fontSize: 12, color: '#475569' }}>
                                      Verification: {JSON.stringify(step.data.verification)}
                                    </div>
                                  )}
                                </div>
                              ))}
                            </div>
                          )}
                        </div>

                        <div style={{ marginTop: 12 }}>
                          <button
                            onClick={() => setOpenTrace(isOpen ? null : idx)}
                            style={{
                              display: 'flex', alignItems: 'center', gap: 6,
                              fontSize: 12, fontFamily: 'monospace', color: '#6366f1',
                              background: 'none', border: 'none', cursor: 'pointer', padding: 0,
                            }}
                          >
                            <Code2 size={13} />
                            {isOpen ? '收起' : '檢視'} FinAgent-RAG 思考軌跡 & PoT 沙盒程式碼
                            {isOpen ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
                          </button>

                          {isOpen && (
                            <div style={{ marginTop: 12 }}>
                              {msg.result.pot_code && (
                                <CodeBlock code={msg.result.pot_code} log={msg.result.sandbox_log} />
                              )}
                              <TriCheckBadge verification={msg.result.verification} />
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/* ─── Loading Indicator ─── */
function LoadingBubble() {
  return (
    <div style={{ display: 'flex', gap: 14, alignItems: 'flex-start' }}>
      <div style={{
        width: 32, height: 32, borderRadius: '50%', background: '#10a37f',
        display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0, marginTop: 2,
      }}>
        <Bot size={16} color="#fff" />
      </div>
      <div style={{
        background: '#fff', border: '1px solid #e5e5e5', borderRadius: 18,
        padding: '16px 22px', display: 'flex', alignItems: 'center', gap: 10,
        fontSize: 13, color: '#10a37f', fontFamily: 'monospace',
        boxShadow: '0 1px 6px rgba(0,0,0,0.04)',
      }}>
        <Sparkles size={14} style={{ animation: 'blink 1s step-end infinite' }} />
        FinAgent-RAG 執行 Query Decompose → Hybrid Retrieval → PoT Sandbox → Tri-Check...
      </div>
    </div>
  );
}

/* ─── Input Bar ─── */
function InputBar({ value, onChange, onSubmit, onFileClick, uploading, disabled }) {
  const textareaRef = useRef(null);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 160) + 'px';
    }
  }, [value]);

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onSubmit();
    }
  };

  return (
    <div style={{
      background: '#fff', border: '1px solid #d9d9e3',
      borderRadius: 20, boxShadow: '0 4px 24px rgba(0,0,0,0.07)',
      padding: '8px 10px 8px 14px',
      display: 'flex', flexDirection: 'column', gap: 8,
    }}>
      {/* Textarea row */}
      <div style={{ display: 'flex', alignItems: 'flex-end', gap: 8 }}>
        <textarea
          ref={textareaRef}
          value={value}
          onChange={onChange}
          onKeyDown={handleKey}
          disabled={disabled}
          placeholder="想問什麼都可以"
          rows={1}
          style={{
            flex: 1, resize: 'none', border: 'none', outline: 'none',
            fontSize: 15, color: '#0d0d0d', background: 'transparent',
            fontFamily: 'Inter, sans-serif', lineHeight: 1.6,
            paddingTop: 4, paddingBottom: 2, maxHeight: 160,
          }}
        />
        {/* Send button */}
        <button
          onClick={onSubmit}
          disabled={disabled || !value.trim()}
          style={{
            width: 36, height: 36, borderRadius: '50%',
            background: value.trim() && !disabled ? '#0d0d0d' : '#e5e5e5',
            border: 'none', cursor: value.trim() && !disabled ? 'pointer' : 'default',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            flexShrink: 0, transition: 'background 0.2s',
          }}
        >
          <Send size={15} color={value.trim() && !disabled ? '#fff' : '#aaa'} />
        </button>
      </div>

      {/* Bottom toolbar row */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 4, paddingLeft: 2 }}>
        {/* File attach button */}
        <button
          onClick={onFileClick}
          disabled={uploading}
          title="上傳財報 PDF / CSV / TXT / JSON"
          style={{
            display: 'flex', alignItems: 'center', gap: 5,
            padding: '4px 10px', borderRadius: 999,
            border: '1px solid #e5e5e5', background: '#f5f5f5',
            cursor: 'pointer', fontSize: 12, color: '#555', fontWeight: 500,
            transition: 'background 0.15s',
          }}
          onMouseEnter={e => e.currentTarget.style.background = '#ececec'}
          onMouseLeave={e => e.currentTarget.style.background = '#f5f5f5'}
        >
          <Plus size={13} />
          {uploading ? '解析中...' : '上傳財報'}
        </button>

        <div style={{ flex: 1 }} />

        <span style={{ fontSize: 11, color: '#c4c4c4', paddingRight: 4 }}>
          Shift + Enter 換行
        </span>
      </div>
    </div>
  );
}

/* ─── Welcome / Empty State ─── */
function WelcomeScreen({ uploadedFiles, onFileClick, uploading }) {
  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 36, padding: '40px 20px 160px' }}>
      {/* Main Headline */}
      <h1 style={{ fontSize: 30, fontWeight: 600, color: '#0d0d0d', textAlign: 'center', letterSpacing: '-0.5px' }}>
        請上傳財報開始提問
      </h1>

      {/* Uploaded files indicator */}
      {uploadedFiles.length > 0 && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, justifyContent: 'center' }} className="fade-in">
          {uploadedFiles.map((f, i) => (
            <span key={i} style={{
              display: 'inline-flex', alignItems: 'center', gap: 6,
              fontSize: 12, padding: '5px 13px', borderRadius: 999,
              background: '#f0fdf4', border: '1px solid #bbf7d0', color: '#166534', fontWeight: 500,
            }}>
              <FileText size={12} />
              {f.filename} · {f.passage_count} Chunks
            </span>
          ))}
        </div>
      )}

      {/* Quick action list */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4, width: '100%', maxWidth: 400 }}>
        {/* Upload row */}
        <button onClick={onFileClick} disabled={uploading} style={{
          display: 'flex', alignItems: 'center', gap: 12,
          padding: '11px 16px', borderRadius: 12, border: '1px solid #e5e5e5',
          background: '#fafafa', cursor: 'pointer', textAlign: 'left',
          transition: 'background 0.15s', width: '100%',
        }}
          onMouseEnter={e => e.currentTarget.style.background = '#f0f0f0'}
          onMouseLeave={e => e.currentTarget.style.background = '#fafafa'}
        >
          <Paperclip size={16} color="#555" />
          <span style={{ fontSize: 14, color: '#333', fontWeight: 500 }}>
            {uploading ? '解析向量中...' : '上傳財報 PDF / CSV / TXT / JSON'}
          </span>
        </button>
      </div>
    </div>
  );
}

/* ─── Main Chat Terminal ─── */
export default function ChatTerminal({
  onQuerySubmit, isLoading, messages,
  samplePrompts, uploadedFiles, onFileUploadSuccess,
}) {
  const [inputQuery, setInputQuery]   = useState('');
  const [uploading,  setUploading]    = useState(false);
  const [uploadMsg,  setUploadMsg]    = useState(null);
  const [openTrace,  setOpenTrace]    = useState(null);
  const fileInputRef                  = useRef(null);
  const bottomRef                     = useRef(null);

  // Auto-scroll
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const handleSubmit = () => {
    if (!inputQuery.trim() || isLoading) return;
    onQuerySubmit(inputQuery.trim());
    setInputQuery('');
  };

  const handleFileChange = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setUploadMsg(null);

    const fd = new FormData();
    fd.append('file', file);
    fd.append('company', file.name.split('.')[0]);

    try {
      const res  = await fetch('http://localhost:8000/api/upload-file', { method: 'POST', body: fd });
      const data = await res.json();

      if (!res.ok) {
        // HTTP error (4xx / 5xx)
        setUploadMsg({ ok: false, warn: false, text: `❌ 上傳失敗: ${data.detail || '未知錯誤'}` });
      } else if (data.passages_added === 0) {
        // Parsed but 0 chunks — likely scanned PDF or empty file
        const warnText = data.warning ||
          `⚠️ "${file.name}" 解析完成，但新增 0 個 Chunk。請上傳可搜尋的 PDF、CSV 或 TXT 檔案。`;
        setUploadMsg({ ok: false, warn: true, text: warnText });
      } else {
        // Success
        setUploadMsg({ ok: true, warn: false, text: `✅ "${file.name}" 已解析成功，新增 ${data.passages_added} 個 Chunk！` });
        onFileUploadSuccess?.();
      }
    } catch (err) {
      setUploadMsg({ ok: false, warn: false, text: `❌ 連線錯誤: ${err.message}` });
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const showWelcome = messages.length === 0 && !isLoading;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', position: 'relative' }}>
      
      {/* Hidden file input */}
      <input type="file" ref={fileInputRef} onChange={handleFileChange}
        accept=".pdf,.csv,.txt,.md,.json" style={{ display: 'none' }} />

      {/* Upload notification banner */}
      {uploadMsg && (
        <div className="fade-in" style={{
          position: 'sticky', top: 0, zIndex: 10,
          margin: '0 auto', maxWidth: 620, width: '100%',
          padding: '10px 18px', marginBottom: 8,
          background: uploadMsg.ok ? '#f0fdf4' : uploadMsg.warn ? '#fffbeb' : '#fff1f2',
          border: `1px solid ${uploadMsg.ok ? '#bbf7d0' : uploadMsg.warn ? '#fde68a' : '#fecdd3'}`,
          borderRadius: 12, fontSize: 13,
          color: uploadMsg.ok ? '#166534' : uploadMsg.warn ? '#92400e' : '#991b1b',
          display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 10,
        }}>
          <span style={{ flex: 1 }}>{uploadMsg.text}</span>
          <button onClick={() => setUploadMsg(null)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#888', padding: 2 }}>
            <X size={14} />
          </button>
        </div>
      )}

      {/* Welcome screen */}
      {showWelcome && (
        <WelcomeScreen
          samplePrompts={samplePrompts}
          uploadedFiles={uploadedFiles}
          onFileClick={() => fileInputRef.current?.click()}
          onPromptClick={(q) => { setInputQuery(q); onQuerySubmit(q); setInputQuery(''); }}
          uploading={uploading}
        />
      )}

      {/* Conversation messages */}
      {messages.length > 0 && (
        <div style={{
          flex: 1, overflowY: 'auto', padding: '32px 0 180px',
        }}>
          <div style={{ maxWidth: 720, margin: '0 auto', padding: '0 24px', display: 'flex', flexDirection: 'column', gap: 36 }}>
            {messages.map((msg, idx) => (
              <MessagePair
                key={idx} msg={msg} idx={idx}
                openTrace={openTrace} setOpenTrace={setOpenTrace}
              />
            ))}
            {isLoading && <LoadingBubble />}
            <div ref={bottomRef} />
          </div>
        </div>
      )}

      {/* Fixed bottom input bar */}
      <div style={{
        position: 'fixed', bottom: 0, left: 0, right: 0,
        background: 'linear-gradient(to top, rgba(255,255,255,1) 70%, rgba(255,255,255,0))',
        padding: '12px 24px 24px',
        zIndex: 40,
      }}>
        <div style={{ maxWidth: 720, margin: '0 auto' }}>
          <InputBar
            value={inputQuery}
            onChange={e => setInputQuery(e.target.value)}
            onSubmit={handleSubmit}
            onFileClick={() => fileInputRef.current?.click()}
            uploading={uploading}
            disabled={isLoading}
          />
          <p style={{ textAlign: 'center', fontSize: 11, color: '#c4c4c4', marginTop: 10 }}>
            FinAgent-RAG 使用確定性 Python AST 沙盒計算財務數據，消除 LLM 心算錯誤。
          </p>
        </div>
      </div>
    </div>
  );
}
