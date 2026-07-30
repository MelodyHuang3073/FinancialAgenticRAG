import React from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend, LineChart, Line } from 'recharts';
import { TrendingUp, DollarSign, PieChart, Activity } from 'lucide-react';

export default function FinancialDashboard({ sampleData }) {
  // TSMC Data
  const tsmcAnnualData = [
    { metric: "營業收入 (Revenue)", "2023 年": 2161.7, "2024 年": 2894.3, YoY: "+33.9%" },
    { metric: "營業毛利 (Gross Profit)", "2023 年": 1175.5, "2024 年": 1652.7, YoY: "+40.6%" },
    { metric: "營業利益 (Operating Income)", "2023 年": 911.2, "2024 年": 1311.5, YoY: "+43.9%" },
    { metric: "本期淨利 (Net Income)", "2023 年": 852.8, "2024 年": 1202.5, YoY: "+41.0%" },
  ];

  const tsmcQuarterlyData = [
    { quarter: "2024 Q1", Revenue: 592.6, NetIncome: 225.5, EPS: 8.70 },
    { quarter: "2024 Q2", Revenue: 673.5, NetIncome: 247.8, EPS: 9.56 },
    { quarter: "2024 Q3", Revenue: 759.7, NetIncome: 325.3, EPS: 12.54 },
    { quarter: "2024 Q4", Revenue: 860.2, NetIncome: 371.7, EPS: 14.33 },
  ];

  // NVIDIA Data
  const nvdaData = [
    { segment: "Data Center", FY2024: 47525, FY2025: 109240 },
    { segment: "Gaming", FY2024: 10447, FY2025: 11850 },
    { segment: "Total Revenue", FY2024: 60922, FY2025: 125984 },
  ];

  return (
    <div className="space-y-6">
      
      {/* Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        
        <div className="glass-panel p-5 space-y-2 border-l-4 border-amber-400">
          <div className="flex items-center justify-between text-xs text-[var(--text-muted)] font-mono">
            <span>台積電 2024 年營收</span>
            <DollarSign className="w-4 h-4 text-amber-400" />
          </div>
          <p className="text-2xl font-bold text-white font-mono">NT$ 2.89 兆</p>
          <p className="text-xs text-emerald-400 flex items-center gap-1 font-mono">
            <TrendingUp className="w-3.5 h-3.5" />
            +33.89% YoY 強勁成長
          </p>
        </div>

        <div className="glass-panel p-5 space-y-2 border-l-4 border-emerald-400">
          <div className="flex items-center justify-between text-xs text-[var(--text-muted)] font-mono">
            <span>台積電 2024 毛利率</span>
            <Activity className="w-4 h-4 text-emerald-400" />
          </div>
          <p className="text-2xl font-bold text-white font-mono">57.1%</p>
          <p className="text-xs text-emerald-400 flex items-center gap-1 font-mono">
            <TrendingUp className="w-3.5 h-3.5" />
            較 2023 年提升 2.7 百分點
          </p>
        </div>

        <div className="glass-panel p-5 space-y-2 border-l-4 border-sky-400">
          <div className="flex items-center justify-between text-xs text-[var(--text-muted)] font-mono">
            <span>台積電 2024 EPS</span>
            <PieChart className="w-4 h-4 text-sky-400" />
          </div>
          <p className="text-2xl font-bold text-white font-mono">NT$ 46.36</p>
          <p className="text-xs text-sky-400 font-mono">
            創歷史新高紀錄
          </p>
        </div>

        <div className="glass-panel p-5 space-y-2 border-l-4 border-purple-400">
          <div className="flex items-center justify-between text-xs text-[var(--text-muted)] font-mono">
            <span>NVIDIA Data Center FY25</span>
            <DollarSign className="w-4 h-4 text-purple-400" />
          </div>
          <p className="text-2xl font-bold text-white font-mono">$109.24 B</p>
          <p className="text-xs text-purple-400 flex items-center gap-1 font-mono">
            <TrendingUp className="w-3.5 h-3.5" />
            +129.8% YoY 爆發成長
          </p>
        </div>

      </div>

      {/* Charts Row 1 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        
        {/* TSMC Annual Comparison */}
        <div className="glass-panel p-6 space-y-4">
          <div>
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-amber-400" />
              台積電 (2330) 2023 vs 2024 損益三表對比 (十億新台幣)
            </h3>
            <p className="text-xs text-[var(--text-muted)]">展示由 FinAgent-RAG 結構化表格引擎對照之關鍵財務指標</p>
          </div>

          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={tsmcAnnualData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a3850" />
                <XAxis dataKey="metric" stroke="#9ca3af" tick={{ fontSize: 11 }} />
                <YAxis stroke="#9ca3af" tick={{ fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: "#121824", borderColor: "#2a3850", borderRadius: "8px" }}
                  itemStyle={{ fontSize: "12px" }}
                />
                <Legend wrapperStyle={{ fontSize: "12px" }} />
                <Bar dataKey="2023 年" fill="#38bdf8" radius={[4, 4, 0, 0]} />
                <Bar dataKey="2024 年" fill="#fbbf24" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* TSMC 2024 Quarterly Trend */}
        <div className="glass-panel p-6 space-y-4">
          <div>
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <Activity className="w-4 h-4 text-emerald-400" />
              台積電 2024 年逐季營收與單季 EPS 趨勢
            </h3>
            <p className="text-xs text-[var(--text-muted)]">N3 / N5 產能利用率滿載帶動營收逐季陡峭上升</p>
          </div>

          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={tsmcQuarterlyData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a3850" />
                <XAxis dataKey="quarter" stroke="#9ca3af" tick={{ fontSize: 11 }} />
                <YAxis yAxisId="left" stroke="#34d399" tick={{ fontSize: 11 }} />
                <YAxis yAxisId="right" orientation="right" stroke="#fbbf24" tick={{ fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: "#121824", borderColor: "#2a3850", borderRadius: "8px" }}
                  itemStyle={{ fontSize: "12px" }}
                />
                <Legend wrapperStyle={{ fontSize: "12px" }} />
                <Line yAxisId="left" type="monotone" dataKey="Revenue" name="營收 (B TWD)" stroke="#34d399" strokeWidth={3} dot={{ r: 5 }} />
                <Line yAxisId="right" type="monotone" dataKey="EPS" name="單季 EPS (TWD)" stroke="#fbbf24" strokeWidth={3} dot={{ r: 5 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

      </div>

      {/* NVIDIA Chart */}
      <div className="glass-panel p-6 space-y-4">
        <div>
          <h3 className="text-base font-bold text-white flex items-center gap-2">
            <DollarSign className="w-4 h-4 text-purple-400" />
            NVIDIA FY2024 vs FY2025 Data Center 營收對比 (Million USD)
          </h3>
          <p className="text-xs text-[var(--text-muted)]">美股 10-K 報告結構化抽取與 Program-of-Thought (PoT) 數值計算展示</p>
        </div>

        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={nvdaData} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a3850" />
              <XAxis dataKey="segment" stroke="#9ca3af" tick={{ fontSize: 11 }} />
              <YAxis stroke="#9ca3af" tick={{ fontSize: 11 }} />
              <Tooltip
                contentStyle={{ backgroundColor: "#121824", borderColor: "#2a3850", borderRadius: "8px" }}
                itemStyle={{ fontSize: "12px" }}
              />
              <Legend wrapperStyle={{ fontSize: "12px" }} />
              <Bar dataKey="FY2024" fill="#a855f7" radius={[4, 4, 0, 0]} />
              <Bar dataKey="FY2025" fill="#38bdf8" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

    </div>
  );
}
