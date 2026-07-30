import React, { useState } from 'react';
import { FileText, Upload, Database, CheckCircle, PlusCircle } from 'lucide-react';

export default function DocumentManager({ sampleData, onUploadSuccess }) {
  const [company, setCompany] = useState('');
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState('');

  const handleUploadSubmit = async (e) => {
    e.preventDefault();
    if (!company || !title || !content) return;

    setIsUploading(true);
    setUploadMessage('');

    try {
      const res = await fetch('http://localhost:8000/api/upload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ company, title, content }),
      });
      const data = await res.json();
      if (res.ok) {
        setUploadMessage(`✅ 成功匯入文件！已向量化建索引，目前語料庫 Chunk 總數: ${data.total_indexed}`);
        setCompany('');
        setTitle('');
        setContent('');
        if (onUploadSuccess) onUploadSuccess();
      } else {
        setUploadMessage(`❌ 匯入失敗: ${data.detail || '伺服器錯誤'}`);
      }
    } catch (err) {
      setUploadMessage(`❌ 連線錯誤: ${err.message}`);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="space-y-6">
      
      {/* Upload Form Panel */}
      <div className="glass-panel p-6 space-y-4 border border-purple-500/30">
        <div className="flex items-center gap-2 text-purple-400">
          <Upload className="w-5 h-5" />
          <h3 className="text-base font-bold text-white">匯入自訂企業財報語料 (Custom Document Ingestion)</h3>
        </div>
        <p className="text-xs text-[var(--text-muted)]">
          上傳自訂企業財報內文、財務三表數據或 MD&A 附註，系統將自動進行 Header-Prepended Row 展平與 Vector Store 向量化索引。
        </p>

        <form onSubmit={handleUploadSubmit} className="space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1">企業名稱 (Company Name)</label>
              <input
                type="text"
                value={company}
                onChange={(e) => setCompany(e.target.value)}
                placeholder="例如：聯發科 (MediaTek 2454)"
                required
                className="w-full bg-[var(--bg-primary)] border border-[var(--border-color)] rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-purple-500"
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1">文件標題 / 報表類型 (Title)</label>
              <input
                type="text"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="例如：2024 年第四季營運報告與展望"
                required
                className="w-full bg-[var(--bg-primary)] border border-[var(--border-color)] rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-purple-500"
              />
            </div>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-300 mb-1">財報內文 / 結構化數據 (Report Text / Financial Content)</label>
            <textarea
              value={content}
              onChange={(e) => setContent(e.target.value)}
              rows={4}
              placeholder="請貼入財報文字、經營討論或結構化數據表格..."
              required
              className="w-full bg-[var(--bg-primary)] border border-[var(--border-color)] rounded-lg p-3 text-sm text-white focus:outline-none focus:border-purple-500 font-sans"
            />
          </div>

          <div className="flex items-center justify-between">
            <button
              type="submit"
              disabled={isUploading}
              className="px-6 py-2.5 bg-gradient-to-r from-purple-500 to-indigo-600 hover:from-purple-400 hover:to-indigo-500 text-white text-sm font-semibold rounded-lg flex items-center gap-2 shadow-lg shadow-purple-500/20"
            >
              {isUploading ? "向量化處置中..." : "新增至語料庫"}
              <PlusCircle className="w-4 h-4" />
            </button>

            {uploadMessage && (
              <span className="text-xs font-mono font-medium text-emerald-400">{uploadMessage}</span>
            )}
          </div>
        </form>
      </div>

      {/* Built-in Corpus Inspector */}
      <div className="glass-panel p-6 space-y-4">
        <div className="flex items-center gap-2 text-amber-400">
          <Database className="w-5 h-5" />
          <h3 className="text-base font-bold text-white">現有內建財報語料庫 (Indexed Knowledge Base)</h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {sampleData.map((doc, idx) => (
            <div key={idx} className="p-4 rounded-xl bg-[var(--bg-secondary)] border border-[var(--border-color)] space-y-3">
              <div className="flex items-center justify-between border-b border-[var(--border-color)] pb-2">
                <span className="font-bold text-sm text-white flex items-center gap-2">
                  <FileText className="w-4 h-4 text-amber-400" />
                  {doc.company}
                </span>
                <span className="text-xs font-mono text-amber-300 px-2 py-0.5 rounded bg-amber-500/10 border border-amber-500/20">
                  {doc.period}
                </span>
              </div>

              <div className="space-y-1.5 text-xs text-[var(--text-muted)]">
                <p className="font-semibold text-slate-300">已索引表格 (Statements):</p>
                <ul className="list-disc pl-4 space-y-0.5">
                  {doc.statements.map((s, si) => (
                    <li key={si}>{s.table_name} ({s.rows.length} 列數據)</li>
                  ))}
                </ul>
                <p className="font-semibold text-slate-300 pt-1">經營討論 (MD&A Notes):</p>
                <ul className="list-disc pl-4 space-y-0.5">
                  {doc.notes.map((n, ni) => (
                    <li key={ni}>{n.title}</li>
                  ))}
                </ul>
              </div>
            </div>
          ))}
        </div>
      </div>

    </div>
  );
}
