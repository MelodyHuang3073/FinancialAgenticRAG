import React from 'react';
import { ChevronDown, RefreshCw, Maximize2, Zap } from 'lucide-react';

export default function Header({ onNewChat, healthStatus, userName = 'Melody' }) {
  return (
    <header style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '12px 20px',
      borderBottom: '1px solid #e5e5e5',
      backgroundColor: '#ffffff',
      position: 'sticky',
      top: 0,
      zIndex: 50,
    }}>
      {/* Left: Brand */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer' }}>
        <span style={{ fontWeight: 700, fontSize: 18, color: '#0d0d0d', letterSpacing: '-0.3px' }}>
          FinAgent RAG
        </span>
        <ChevronDown size={16} color="#666" />
      </div>

      {/* Right: Actions */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        {/* Engine Status */}
        <span style={{
          display: 'flex',
          alignItems: 'center',
          gap: 5,
          fontSize: 12,
          fontWeight: 500,
          color: healthStatus ? '#16a34a' : '#888',
          backgroundColor: healthStatus ? '#f0fdf4' : '#f5f5f5',
          border: `1px solid ${healthStatus ? '#bbf7d0' : '#e5e5e5'}`,
          borderRadius: 999,
          padding: '4px 12px',
        }}>
          <Zap size={12} fill={healthStatus ? '#16a34a' : '#aaa'} color={healthStatus ? '#16a34a' : '#aaa'} />
          {healthStatus ? 'PoT 引擎就緒' : '連線中...'}
        </span>

        {/* New Chat */}
        <button onClick={onNewChat} title="清除對話" style={{
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          width: 34, height: 34, borderRadius: 8,
          border: '1px solid #e5e5e5', background: '#fff', cursor: 'pointer',
          color: '#555', transition: 'background 0.15s',
        }}
          onMouseEnter={e => e.currentTarget.style.background = '#f5f5f5'}
          onMouseLeave={e => e.currentTarget.style.background = '#fff'}
        >
          <RefreshCw size={15} />
        </button>

        {/* Fullscreen */}
        <button onClick={() => {
          if (!document.fullscreenElement) document.documentElement.requestFullscreen();
          else document.exitFullscreen();
        }} title="全螢幕" style={{
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          width: 34, height: 34, borderRadius: 8,
          border: '1px solid #e5e5e5', background: '#fff', cursor: 'pointer',
          color: '#555', transition: 'background 0.15s',
        }}
          onMouseEnter={e => e.currentTarget.style.background = '#f5f5f5'}
          onMouseLeave={e => e.currentTarget.style.background = '#fff'}
        >
          <Maximize2 size={15} />
        </button>
      </div>
    </header>
  );
}
