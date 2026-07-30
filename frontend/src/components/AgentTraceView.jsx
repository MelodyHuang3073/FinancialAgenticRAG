import React from 'react';
import { Sparkles, GitBranch, Cpu, ShieldCheck, CheckCircle2, XCircle, Code2, Layers, Search, RefreshCw } from 'lucide-react';

export default function AgentTraceView({ result }) {
  if (!result || !result.execution_trace) {
    return (
      <div className="glass-panel p-12 text-center text-[var(--text-muted)] space-y-3">
        <Sparkles className="w-12 h-12 text-amber-400 mx-auto opacity-50" />
        <h3 className="text-base font-semibold text-white">尚無 Agent 思考軌跡記錄</h3>
        <p className="text-xs">請先在「問答對話終端」提出財報問題，系統將在此呈現完整的 Agentic Refinement Trace。</p>
      </div>
    );
  }

  const { execution_trace, query, complexity, total_iterations, verification } = result;

  return (
    <div className="space-y-6">
      
      {/* Top Banner Summary */}
      <div className="glass-panel p-6 border-l-4 border-amber-500 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <span className="text-xs font-mono uppercase px-2.5 py-1 rounded bg-amber-500/10 text-amber-400 border border-amber-500/20">
            FinAgent-RAG Execution Trace Audit
          </span>
          <h2 className="text-lg font-bold text-white mt-2">{query}</h2>
        </div>

        <div className="flex items-center gap-4 text-xs font-mono">
          <div className="bg-[var(--bg-secondary)] px-3 py-2 rounded-lg border border-[var(--border-color)]">
            <span className="text-[var(--text-muted)] block">複雜度類別</span>
            <span className="text-amber-400 font-bold">{complexity}</span>
          </div>
          <div className="bg-[var(--bg-secondary)] px-3 py-2 rounded-lg border border-[var(--border-color)]">
            <span className="text-[var(--text-muted)] block">總迭代次數</span>
            <span className="text-sky-400 font-bold">{total_iterations} / 3</span>
          </div>
          <div className="bg-[var(--bg-secondary)] px-3 py-2 rounded-lg border border-[var(--border-color)]">
            <span className="text-[var(--text-muted)] block">Tri-Check 狀態</span>
            <span className={verification?.decision === "ACCEPT" ? "text-emerald-400 font-bold" : "text-rose-400 font-bold"}>
              {verification?.decision} ({Math.round((verification?.confidence_score || 0) * 100)}%)
            </span>
          </div>
        </div>
      </div>

      {/* Timeline Steps */}
      <div className="space-y-4 relative before:absolute before:inset-0 before:left-6 before:w-0.5 before:bg-[var(--border-color)]">
        {execution_trace.map((step, index) => (
          <div key={index} className="relative pl-12">
            
            {/* Step Icon Indicator */}
            <div className="absolute left-2.5 top-1.5 -translate-x-1/2 w-8 h-8 rounded-full bg-[var(--bg-secondary)] border-2 border-amber-500 flex items-center justify-center text-amber-400 text-xs font-bold shadow-md">
              {index + 1}
            </div>

            {/* Step Card */}
            <div className="glass-panel p-5 space-y-3">
              <div className="flex items-center justify-between">
                <h4 className="text-sm font-bold text-white flex items-center gap-2">
                  {step.type === 'routing' && <Cpu className="w-4 h-4 text-amber-400" />}
                  {step.type === 'decomposition' && <GitBranch className="w-4 h-4 text-sky-400" />}
                  {step.type === 'iteration' && <Layers className="w-4 h-4 text-emerald-400" />}
                  {step.type === 'refinement' && <RefreshCw className="w-4 h-4 text-purple-400" />}
                  {step.step_name}
                </h4>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-[var(--bg-primary)] text-[var(--text-muted)] border border-[var(--border-color)] uppercase">
                  {step.type}
                </span>
              </div>

              {/* Step Detail Text */}
              {step.detail && (
                <p className="text-xs text-slate-300 font-sans">{step.detail}</p>
              )}

              {/* Sub questions for Decomposition */}
              {step.sub_questions && (
                <div className="space-y-1.5 pt-2">
                  <h5 className="text-xs font-semibold text-sky-400">分解子任務 (Sub-Questions Tree):</h5>
                  <div className="space-y-1 pl-2 border-l-2 border-sky-500/30">
                    {step.sub_questions.map((sq, i) => (
                      <div key={i} className="text-xs text-slate-300 font-mono flex items-center gap-2">
                        <span className="text-sky-400">Step {sq.step}:</span>
                        <span>{sq.query}</span>
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-sky-500/10 text-sky-300">[{sq.type}]</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Data payload for Iteration step */}
              {step.data && (
                <div className="space-y-3 pt-2">
                  
                  {/* Retrieved Passages */}
                  {step.data.retrieved_passages && step.data.retrieved_passages.length > 0 && (
                    <div className="space-y-1">
                      <h5 className="text-xs font-semibold text-emerald-400 flex items-center gap-1">
                        <Search className="w-3.5 h-3.5" />
                        檢索到之語料 Chunk (Top Hits):
                      </h5>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                        {step.data.retrieved_passages.map((p, pi) => (
                          <div key={pi} className="p-2.5 rounded bg-[var(--bg-primary)] border border-[var(--border-color)] text-[11px] space-y-1">
                            <div className="flex justify-between text-amber-300 font-mono">
                              <span>{p.table_name}</span>
                              <span className="text-sky-400">{p.relevance_score}</span>
                            </div>
                            <p className="text-[var(--text-muted)] line-clamp-2">{p.snippet}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Sandboxed PoT Python Code */}
                  {step.data.pot_code && (
                    <div className="space-y-1">
                      <h5 className="text-xs font-semibold text-purple-400 flex items-center gap-1 font-mono">
                        <Code2 className="w-3.5 h-3.5" />
                        Program-of-Thought (PoT) Python Sandboxed Execution:
                      </h5>
                      <pre className="text-[11px] font-mono text-emerald-300 bg-slate-950 p-3 rounded border border-slate-800 overflow-x-auto leading-relaxed">
                        {step.data.pot_code}
                      </pre>
                      {step.data.sandbox_output && (
                        <div className="text-xs font-mono text-amber-300 bg-amber-500/10 p-2 rounded border border-amber-500/20">
                          Execution Output: {step.data.sandbox_output}
                        </div>
                      )}
                    </div>
                  )}

                  {/* Tri-Check Verification Badges */}
                  {step.data.verification && step.data.verification.checks && (
                    <div className="pt-2 border-t border-[var(--border-color)] space-y-2">
                      <h5 className="text-xs font-semibold text-amber-400 flex items-center gap-1">
                        <ShieldCheck className="w-3.5 h-3.5" />
                        Tri-Check Self-Verification Checks ($\nu_{{suff}}, \nu_{{num}}, \nu_{{cross}}$):
                      </h5>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-2 text-xs">
                        
                        {/* Sufficiency */}
                        <div className="p-2.5 rounded bg-[var(--bg-primary)] border border-[var(--border-color)] space-y-1">
                          <div className="flex items-center justify-between">
                            <span className="font-mono text-amber-300 font-semibold">$\nu_{{suff}}$ (數據充分)</span>
                            {step.data.verification.checks.nu_suff.passed ? (
                              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                            ) : (
                              <XCircle className="w-4 h-4 text-rose-400" />
                            )}
                          </div>
                          <p className="text-[11px] text-[var(--text-muted)]">{step.data.verification.checks.nu_suff.detail}</p>
                        </div>

                        {/* Numerical */}
                        <div className="p-2.5 rounded bg-[var(--bg-primary)] border border-[var(--border-color)] space-y-1">
                          <div className="flex items-center justify-between">
                            <span className="font-mono text-amber-300 font-semibold">$\nu_{{num}}$ (數值算術)</span>
                            {step.data.verification.checks.nu_num.passed ? (
                              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                            ) : (
                              <XCircle className="w-4 h-4 text-rose-400" />
                            )}
                          </div>
                          <p className="text-[11px] text-[var(--text-muted)]">{step.data.verification.checks.nu_num.detail}</p>
                        </div>

                        {/* Cross Evidence */}
                        <div className="p-2.5 rounded bg-[var(--bg-primary)] border border-[var(--border-color)] space-y-1">
                          <div className="flex items-center justify-between">
                            <span className="font-mono text-amber-300 font-semibold">$\nu_{{cross}}$ (跨期脈絡)</span>
                            {step.data.verification.checks.nu_cross.passed ? (
                              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                            ) : (
                              <XCircle className="w-4 h-4 text-rose-400" />
                            )}
                          </div>
                          <p className="text-[11px] text-[var(--text-muted)]">{step.data.verification.checks.nu_cross.detail}</p>
                        </div>

                      </div>
                    </div>
                  )}

                </div>
              )}

            </div>
          </div>
        ))}
      </div>

    </div>
  );
}
