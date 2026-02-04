'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { useAuthStore } from '@/lib/store/authStore';
import { getUserProfile, type UserProfile } from '@/lib/api/auth';
import { authFetch } from '@/lib/store/authStore';
import { NavHeader } from '@/components/NavHeader';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface Scaffold {
  id: number;
  uuid: string;
  name: string;
  scaffold_type: string;
  parameters: Record<string, any>;
  thumbnail_path: string | null;
  created_at: string;
  updated_at: string;
}

interface ScaffoldStats {
  total: number;
  this_week: number;
  most_used_type: string;
}

const SCAFFOLD_TYPES = [
  {
    id: 'gyroid',
    name: 'Gyroid',
    description: 'Triple periodic minimal surface',
    gradient: 'from-cyan-500 to-blue-600'
  },
  {
    id: 'lattice',
    name: 'Lattice',
    description: 'Regular repeating unit cells',
    gradient: 'from-emerald-500 to-teal-600'
  },
  {
    id: 'vascular',
    name: 'Vascular',
    description: 'Biomimetic vessel networks',
    gradient: 'from-rose-500 to-red-600'
  },
  {
    id: 'porous_disc',
    name: 'Porous Disc',
    description: 'Controlled porosity discs',
    gradient: 'from-violet-500 to-purple-600'
  },
  {
    id: 'tubular',
    name: 'Tubular',
    description: 'Cylindrical tissue constructs',
    gradient: 'from-amber-500 to-orange-600'
  },
];

const TIPS = [
  'Use the chat assistant to quickly adjust parameters',
  'Export both binary and ASCII STL formats for compatibility',
  'Save your scaffolds to build a personal library',
  'Check validation warnings before exporting',
  'Experiment with different scaffold types for optimal results',
];

export default function DashboardPage() {
  const router = useRouter();
  const { isAuthenticated, user } = useAuthStore();
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [scaffolds, setScaffolds] = useState<Scaffold[]>([]);
  const [stats, setStats] = useState<ScaffoldStats>({ total: 0, this_week: 0, most_used_type: 'None' });
  const [loading, setLoading] = useState(true);
  const [tipsExpanded, setTipsExpanded] = useState(false);
  const [currentTipIndex, setCurrentTipIndex] = useState(0);

  useEffect(() => {
    if (!isAuthenticated) {
      router.push('/login');
      return;
    }

    const loadData = async () => {
      try {
        // Load profile
        const profileData = await getUserProfile();
        setProfile(profileData);

        // Load scaffolds
        const response = await authFetch(`${API_BASE}/api/scaffolds`);
        if (response.ok) {
          const data = await response.json();
          setScaffolds(data.scaffolds || []);

          // Calculate stats
          const total = data.total || 0;
          const oneWeekAgo = new Date();
          oneWeekAgo.setDate(oneWeekAgo.getDate() - 7);

          const thisWeek = (data.scaffolds || []).filter((s: Scaffold) =>
            new Date(s.created_at) > oneWeekAgo
          ).length;

          // Find most used type
          const typeCounts: Record<string, number> = {};
          (data.scaffolds || []).forEach((s: Scaffold) => {
            typeCounts[s.scaffold_type] = (typeCounts[s.scaffold_type] || 0) + 1;
          });
          const mostUsedType = Object.keys(typeCounts).length > 0
            ? Object.entries(typeCounts).sort((a, b) => b[1] - a[1])[0][0]
            : 'None';

          setStats({ total, this_week: thisWeek, most_used_type: mostUsedType });
        }
      } catch (error) {
        console.error('Failed to load dashboard data:', error);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [isAuthenticated, router]);

  // Rotate tips every 5 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentTipIndex((prev) => (prev + 1) % TIPS.length);
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  };

  const currentDate = new Date().toLocaleDateString('en-US', {
    weekday: 'long',
    year: 'numeric',
    month: 'long',
    day: 'numeric'
  });

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-emerald-950 flex items-center justify-center">
        <div className="text-emerald-400 text-xl font-mono">LOADING DASHBOARD...</div>
      </div>
    );
  }

  const displayName = profile?.display_name || user?.username || 'User';

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-emerald-950">
      <NavHeader currentPage="dashboard" />

      <main className="max-w-7xl mx-auto px-6 py-12 space-y-12">
        {/* Welcome Section */}
        <section className="space-y-6">
          <div className="space-y-2">
            <h1 className="text-5xl font-black text-white tracking-tight">
              Welcome back, <span className="text-emerald-400">{displayName}</span>
            </h1>
            <p className="text-slate-400 font-mono text-sm">{currentDate}</p>
          </div>

          {/* Stats Row */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-slate-900/50 border border-emerald-900/30 p-6 backdrop-blur-sm hover:border-emerald-500/50 transition-colors">
              <div className="text-6xl font-black text-emerald-400 mb-2">{stats.total}</div>
              <div className="text-slate-400 font-mono text-xs tracking-widest">TOTAL SCAFFOLDS</div>
            </div>
            <div className="bg-slate-900/50 border border-cyan-900/30 p-6 backdrop-blur-sm hover:border-cyan-500/50 transition-colors">
              <div className="text-6xl font-black text-cyan-400 mb-2">{stats.this_week}</div>
              <div className="text-slate-400 font-mono text-xs tracking-widest">THIS WEEK</div>
            </div>
            <div className="bg-slate-900/50 border border-violet-900/30 p-6 backdrop-blur-sm hover:border-violet-500/50 transition-colors">
              <div className="text-2xl font-black text-violet-400 mb-2 uppercase">{stats.most_used_type.replace('_', ' ')}</div>
              <div className="text-slate-400 font-mono text-xs tracking-widest">MOST USED TYPE</div>
            </div>
          </div>
        </section>

        {/* Quick Actions */}
        <section className="space-y-6">
          <h2 className="text-2xl font-black text-white tracking-tight">QUICK ACTIONS</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <Link
              href="/generator"
              className="group relative overflow-hidden bg-gradient-to-br from-emerald-600 to-emerald-700 p-8 border-2 border-emerald-400 hover:border-emerald-300 transition-all hover:scale-105 hover:shadow-2xl hover:shadow-emerald-500/50"
            >
              <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent"></div>
              <div className="relative z-10">
                <div className="text-4xl mb-3">🧬</div>
                <div className="text-xl font-black text-white mb-1">New Scaffold</div>
                <div className="text-emerald-100 text-sm font-mono">Generate →</div>
              </div>
            </Link>

            <Link
              href="/library"
              className="group relative overflow-hidden bg-gradient-to-br from-cyan-600 to-cyan-700 p-8 border-2 border-cyan-400 hover:border-cyan-300 transition-all hover:scale-105 hover:shadow-2xl hover:shadow-cyan-500/50"
            >
              <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent"></div>
              <div className="relative z-10">
                <div className="text-4xl mb-3">📚</div>
                <div className="text-xl font-black text-white mb-1">Browse Library</div>
                <div className="text-cyan-100 text-sm font-mono">View All →</div>
              </div>
            </Link>

            <Link
              href="/settings"
              className="group relative overflow-hidden bg-gradient-to-br from-violet-600 to-violet-700 p-8 border-2 border-violet-400 hover:border-violet-300 transition-all hover:scale-105 hover:shadow-2xl hover:shadow-violet-500/50"
            >
              <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent"></div>
              <div className="relative z-10">
                <div className="text-4xl mb-3">⚙️</div>
                <div className="text-xl font-black text-white mb-1">Settings</div>
                <div className="text-violet-100 text-sm font-mono">Configure →</div>
              </div>
            </Link>

            <a
              href="https://github.com/morphostruct/docs"
              target="_blank"
              rel="noopener noreferrer"
              className="group relative overflow-hidden bg-gradient-to-br from-slate-700 to-slate-800 p-8 border-2 border-slate-500 hover:border-slate-400 transition-all hover:scale-105 hover:shadow-2xl hover:shadow-slate-500/50"
            >
              <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent"></div>
              <div className="relative z-10">
                <div className="text-4xl mb-3">📖</div>
                <div className="text-xl font-black text-white mb-1">Help & Docs</div>
                <div className="text-slate-300 text-sm font-mono">Learn →</div>
              </div>
            </a>
          </div>
        </section>

        {/* Recent Scaffolds */}
        {scaffolds.length > 0 && (
          <section className="space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-2xl font-black text-white tracking-tight">RECENT SCAFFOLDS</h2>
              <Link
                href="/library"
                className="text-emerald-400 hover:text-emerald-300 font-mono text-sm transition-colors"
              >
                VIEW ALL →
              </Link>
            </div>
            <div className="flex gap-4 overflow-x-auto pb-4 scrollbar-thin scrollbar-thumb-emerald-900 scrollbar-track-slate-900">
              {scaffolds.slice(0, 5).map((scaffold) => (
                <Link
                  key={scaffold.uuid}
                  href={`/library?scaffold=${scaffold.uuid}`}
                  className="flex-shrink-0 w-64 bg-slate-900/50 border border-slate-700 hover:border-emerald-500/50 transition-all hover:scale-105 overflow-hidden group"
                >
                  <div className="h-48 bg-gradient-to-br from-slate-800 to-slate-900 flex items-center justify-center relative overflow-hidden">
                    {scaffold.thumbnail_path ? (
                      <img
                        src={scaffold.thumbnail_path}
                        alt={scaffold.name}
                        className="w-full h-full object-cover group-hover:scale-110 transition-transform"
                      />
                    ) : (
                      <div className="text-6xl opacity-30">🧬</div>
                    )}
                    <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent"></div>
                  </div>
                  <div className="p-4">
                    <div className="text-white font-bold mb-1 truncate">{scaffold.name}</div>
                    <div className="text-emerald-400 text-xs font-mono uppercase mb-2">
                      {scaffold.scaffold_type.replace('_', ' ')}
                    </div>
                    <div className="text-slate-500 text-xs font-mono">
                      {formatDate(scaffold.created_at)}
                    </div>
                  </div>
                </Link>
              ))}
            </div>
          </section>
        )}

        {/* Scaffold Type Quick Picker */}
        <section className="space-y-6">
          <h2 className="text-2xl font-black text-white tracking-tight">QUICK START</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
            {SCAFFOLD_TYPES.map((type) => (
              <Link
                key={type.id}
                href={`/generator?type=${type.id}`}
                className={`group relative overflow-hidden bg-gradient-to-br ${type.gradient} p-6 border-2 border-white/20 hover:border-white/50 transition-all hover:scale-105`}
              >
                <div className="absolute inset-0 bg-black/20 group-hover:bg-black/10 transition-colors"></div>
                <div className="relative z-10">
                  <div className="text-xl font-black text-white mb-2">{type.name}</div>
                  <div className="text-white/80 text-xs">{type.description}</div>
                </div>
              </Link>
            ))}
          </div>
        </section>

        {/* Tips Section */}
        <section className="space-y-4">
          <button
            onClick={() => setTipsExpanded(!tipsExpanded)}
            className="flex items-center justify-between w-full text-left group"
          >
            <h2 className="text-2xl font-black text-white tracking-tight">TIPS & TRICKS</h2>
            <div className="text-emerald-400 font-mono text-sm">
              {tipsExpanded ? '▼' : '▶'}
            </div>
          </button>

          {tipsExpanded ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {TIPS.map((tip, index) => (
                <div
                  key={index}
                  className="bg-slate-900/50 border border-slate-700 p-6 backdrop-blur-sm"
                >
                  <div className="flex items-start gap-4">
                    <div className="text-emerald-400 font-black text-xl">{index + 1}</div>
                    <div className="text-slate-300 text-sm">{tip}</div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="bg-slate-900/50 border border-emerald-900/30 p-6 backdrop-blur-sm">
              <div className="flex items-start gap-4">
                <div className="text-emerald-400 font-black text-xl">💡</div>
                <div className="text-slate-300 text-sm">{TIPS[currentTipIndex]}</div>
              </div>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
