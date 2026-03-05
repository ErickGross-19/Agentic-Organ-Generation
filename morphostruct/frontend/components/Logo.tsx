'use client';

interface LogoProps {
  size?: number;
  className?: string;
  showText?: boolean;
}

export function Logo({ size = 40, className = '', showText = true }: LogoProps) {
  return (
    <div className={`flex items-center gap-3 ${className}`}>
      <svg
        width={size}
        height={size}
        viewBox="0 0 100 100"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="flex-shrink-0"
      >
        {/* Background glow */}
        <defs>
          <linearGradient id="logoGradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#10b981" />
            <stop offset="50%" stopColor="#34d399" />
            <stop offset="100%" stopColor="#6ee7b7" />
          </linearGradient>
          <linearGradient id="logoGradientDark" x1="0%" y1="100%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#059669" />
            <stop offset="100%" stopColor="#10b981" />
          </linearGradient>
          <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
            <feMerge>
              <feMergeNode in="coloredBlur"/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>
        </defs>

        {/* Outer hexagonal frame - geometric */}
        <path
          d="M50 5 L88 27.5 L88 72.5 L50 95 L12 72.5 L12 27.5 Z"
          stroke="url(#logoGradient)"
          strokeWidth="2"
          fill="none"
          opacity="0.6"
        />

        {/* Inner scaffold lattice - geometric/scientific */}
        <g stroke="url(#logoGradient)" strokeWidth="1.5" opacity="0.4">
          {/* Horizontal struts */}
          <line x1="25" y1="35" x2="75" y2="35" />
          <line x1="20" y1="50" x2="80" y2="50" />
          <line x1="25" y1="65" x2="75" y2="65" />

          {/* Vertical connections */}
          <line x1="35" y1="25" x2="35" y2="75" />
          <line x1="50" y1="20" x2="50" y2="80" />
          <line x1="65" y1="25" x2="65" y2="75" />
        </g>

        {/* Central organic cell structure - biological */}
        <g filter="url(#glow)">
          {/* Main cell body */}
          <ellipse
            cx="50"
            cy="50"
            rx="18"
            ry="16"
            fill="url(#logoGradientDark)"
            opacity="0.9"
          />

          {/* Cell membrane highlight */}
          <ellipse
            cx="50"
            cy="50"
            rx="18"
            ry="16"
            stroke="url(#logoGradient)"
            strokeWidth="2"
            fill="none"
          />

          {/* Nucleus */}
          <circle
            cx="50"
            cy="50"
            r="6"
            fill="#000"
            opacity="0.6"
          />
          <circle
            cx="50"
            cy="50"
            r="6"
            stroke="#6ee7b7"
            strokeWidth="1"
            fill="none"
          />
        </g>

        {/* Branching vessel network - biological/organic */}
        <g stroke="url(#logoGradient)" strokeWidth="2" strokeLinecap="round" fill="none">
          {/* Top branch */}
          <path d="M50 34 Q45 28 38 22" opacity="0.8" />
          <path d="M50 34 Q55 28 62 22" opacity="0.8" />

          {/* Bottom branch */}
          <path d="M50 66 Q45 72 38 78" opacity="0.8" />
          <path d="M50 66 Q55 72 62 78" opacity="0.8" />

          {/* Side branches */}
          <path d="M32 50 Q26 45 20 42" opacity="0.6" />
          <path d="M32 50 Q26 55 20 58" opacity="0.6" />
          <path d="M68 50 Q74 45 80 42" opacity="0.6" />
          <path d="M68 50 Q74 55 80 58" opacity="0.6" />
        </g>

        {/* Connection nodes - sleek tech feel */}
        <g fill="#10b981">
          <circle cx="38" cy="22" r="2.5" />
          <circle cx="62" cy="22" r="2.5" />
          <circle cx="38" cy="78" r="2.5" />
          <circle cx="62" cy="78" r="2.5" />
          <circle cx="20" cy="42" r="2" opacity="0.7" />
          <circle cx="20" cy="58" r="2" opacity="0.7" />
          <circle cx="80" cy="42" r="2" opacity="0.7" />
          <circle cx="80" cy="58" r="2" opacity="0.7" />
        </g>

        {/* Lattice intersection points */}
        <g fill="#34d399" opacity="0.6">
          <circle cx="35" cy="35" r="2" />
          <circle cx="50" cy="35" r="2" />
          <circle cx="65" cy="35" r="2" />
          <circle cx="35" cy="65" r="2" />
          <circle cx="50" cy="65" r="2" />
          <circle cx="65" cy="65" r="2" />
        </g>
      </svg>

      {showText && (
        <div className="flex flex-col">
          <span
            className="text-xl font-black tracking-tight text-white"
            style={{ fontFamily: 'system-ui' }}
          >
            MorphoStruct
          </span>
          <span className="text-[10px] font-mono text-emerald-400/70 tracking-widest">
            BIOPRINTING SCAFFOLD DESIGN
          </span>
        </div>
      )}
    </div>
  );
}

export function LogoIcon({ size = 32, className = '' }: { size?: number; className?: string }) {
  return <Logo size={size} className={className} showText={false} />;
}
