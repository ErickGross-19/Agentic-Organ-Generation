'use client';

import { useState, useEffect, useMemo } from 'react';
import { useRouter } from 'next/navigation';
import { useAuthStore } from '@/lib/store/authStore';
import { listScaffolds, getScaffold, deleteScaffold, duplicateScaffold, SavedScaffold } from '@/lib/api/scaffolds';
import { NavHeader } from '@/components/NavHeader';

type ViewMode = 'grid' | 'list';
type SortMode = 'newest' | 'oldest' | 'name-asc' | 'name-desc' | 'type';

export default function LibraryPage() {
  const router = useRouter();
  const { user } = useAuthStore();
  const [scaffolds, setScaffolds] = useState<SavedScaffold[]>([]);
  const [loading, setLoading] = useState(true);
  const [viewMode, setViewMode] = useState<ViewMode>('grid');
  const [sortMode, setSortMode] = useState<SortMode>('newest');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedTypes, setSelectedTypes] = useState<Set<string>>(new Set());
  const [selectedScaffolds, setSelectedScaffolds] = useState<Set<string>>(new Set());
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [scaffoldToDelete, setScaffoldToDelete] = useState<string | null>(null);
  const [message, setMessage] = useState('');

  useEffect(() => {
    loadScaffolds();
  }, []);

  const loadScaffolds = async () => {
    try {
      const result = await listScaffolds();
      setScaffolds(result.scaffolds);
    } catch (error) {
      console.error('Failed to load scaffolds:', error);
      showMessage('FAILED TO LOAD SCAFFOLDS');
    } finally {
      setLoading(false);
    }
  };

  const showMessage = (msg: string) => {
    setMessage(msg);
    setTimeout(() => setMessage(''), 3000);
  };

  // Filter and sort scaffolds
  const filteredScaffolds = useMemo(() => {
    let filtered = scaffolds;

    // Search filter
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(s =>
        s.name.toLowerCase().includes(query) ||
        s.type.toLowerCase().includes(query)
      );
    }

    // Type filter
    if (selectedTypes.size > 0) {
      filtered = filtered.filter(s => selectedTypes.has(s.type));
    }

    // Sort
    filtered = [...filtered].sort((a, b) => {
      switch (sortMode) {
        case 'newest':
          return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
        case 'oldest':
          return new Date(a.created_at).getTime() - new Date(b.created_at).getTime();
        case 'name-asc':
          return a.name.localeCompare(b.name);
        case 'name-desc':
          return b.name.localeCompare(a.name);
        case 'type':
          return a.type.localeCompare(b.type);
        default:
          return 0;
      }
    });

    return filtered;
  }, [scaffolds, searchQuery, selectedTypes, sortMode]);

  const availableTypes = useMemo(() => {
    return Array.from(new Set(scaffolds.map(s => s.type)));
  }, [scaffolds]);

  const handleLoad = (uuid: string) => {
    router.push(`/generator?scaffold=${uuid}`);
  };

  const handleDuplicate = async (uuid: string) => {
    try {
      await duplicateScaffold(uuid);
      showMessage('SCAFFOLD DUPLICATED');
      loadScaffolds();
    } catch (error) {
      console.error('Failed to duplicate:', error);
      showMessage('DUPLICATION FAILED');
    }
  };

  const handleDelete = async (uuid: string) => {
    setScaffoldToDelete(uuid);
    setShowDeleteModal(true);
  };

  const confirmDelete = async () => {
    if (!scaffoldToDelete) return;

    try {
      await deleteScaffold(scaffoldToDelete);
      showMessage('SCAFFOLD DELETED');
      loadScaffolds();
    } catch (error) {
      console.error('Failed to delete:', error);
      showMessage('DELETION FAILED');
    } finally {
      setShowDeleteModal(false);
      setScaffoldToDelete(null);
    }
  };

  const handleBulkDelete = async () => {
    if (!confirm(`DELETE ${selectedScaffolds.size} SCAFFOLDS?`)) return;

    try {
      await Promise.all(
        Array.from(selectedScaffolds).map(uuid => deleteScaffold(uuid))
      );
      showMessage(`${selectedScaffolds.size} SCAFFOLDS DELETED`);
      setSelectedScaffolds(new Set());
      loadScaffolds();
    } catch (error) {
      console.error('Failed to delete scaffolds:', error);
      showMessage('BULK DELETION FAILED');
    }
  };

  const toggleScaffoldSelection = (uuid: string) => {
    setSelectedScaffolds(prev => {
      const next = new Set(prev);
      if (next.has(uuid)) {
        next.delete(uuid);
      } else {
        next.add(uuid);
      }
      return next;
    });
  };

  const toggleTypeFilter = (type: string) => {
    setSelectedTypes(prev => {
      const next = new Set(prev);
      if (next.has(type)) {
        next.delete(type);
      } else {
        next.add(type);
      }
      return next;
    });
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center">
        <div className="relative">
          <div className="w-16 h-16 border-4 border-white/20 border-t-white rounded-full animate-spin" />
          <div className="absolute inset-0 w-16 h-16 border-4 border-transparent border-l-emerald-400/50 rounded-full animate-spin" style={{ animationDuration: '1.5s' }} />
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-black text-white relative overflow-hidden">
      <NavHeader currentPage="library" />

      {/* Cellular background pattern */}
      <div className="fixed inset-0 opacity-[0.02] pointer-events-none" style={{
        backgroundImage: `radial-gradient(circle at 20% 50%, white 1px, transparent 1px),
                         radial-gradient(circle at 80% 80%, white 1px, transparent 1px),
                         radial-gradient(circle at 40% 20%, white 1px, transparent 1px)`,
        backgroundSize: '100px 100px, 150px 150px, 80px 80px'
      }} />

      {/* Grid overlay */}
      <div className="fixed inset-0 opacity-[0.03] pointer-events-none" style={{
        backgroundImage: 'linear-gradient(rgba(255,255,255,0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.05) 1px, transparent 1px)',
        backgroundSize: '50px 50px'
      }} />

      <div className="relative max-w-7xl mx-auto px-8 py-12">
        {/* Page Title */}
        <div className="mb-12 pb-8 border-b-4 border-white/10">
          <div className="text-xs font-mono text-emerald-400 tracking-widest mb-4">SCAFFOLD_ARCHIVE</div>
          <h1 className="text-6xl font-black tracking-tighter mb-4" style={{ fontFamily: 'system-ui' }}>
            LIBRARY
          </h1>
          <div className="flex items-center gap-4 text-sm font-mono">
            <span className="text-white/40">TOTAL SCAFFOLDS:</span>
            <span className="text-white font-bold">{scaffolds.length}</span>
            <span className="text-white/40">SHOWING:</span>
            <span className="text-emerald-400 font-bold">{filteredScaffolds.length}</span>
          </div>
        </div>

        {/* Status message */}
        {message && (
          <div className="mb-8 p-4 bg-emerald-400/10 border-l-4 border-emerald-400 font-mono text-sm tracking-wide animate-pulse">
            <span className="text-emerald-400">▶</span> {message}
          </div>
        )}

        {/* Controls bar */}
        <div className="mb-8 flex flex-wrap gap-4 items-center justify-between bg-white/[0.02] border-2 border-white/10 p-6">
          {/* Search */}
          <div className="flex-1 min-w-[300px]">
            <div className="relative">
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="SEARCH SCAFFOLDS..."
                className="w-full px-4 py-3 pl-12 bg-black border-2 border-white/20 text-white placeholder-white/30 focus:outline-none focus:border-emerald-400 font-mono text-sm transition-colors"
              />
              <svg className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-white/40" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
            </div>
          </div>

          {/* View mode */}
          <div className="flex gap-1 border-2 border-white/20">
            <button
              onClick={() => setViewMode('grid')}
              className={`px-4 py-3 font-bold text-sm transition-all ${
                viewMode === 'grid' ? 'bg-white text-black' : 'bg-transparent text-white/40 hover:text-white'
              }`}
            >
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                <path d="M10 10H4V4h6v6zm10 0h-6V4h6v6zM10 20H4v-6h6v6zm10 0h-6v-6h6v6z" />
              </svg>
            </button>
            <button
              onClick={() => setViewMode('list')}
              className={`px-4 py-3 font-bold text-sm transition-all ${
                viewMode === 'list' ? 'bg-white text-black' : 'bg-transparent text-white/40 hover:text-white'
              }`}
            >
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                <path d="M4 6h16v2H4zm0 5h16v2H4zm0 5h16v2H4z" />
              </svg>
            </button>
          </div>

          {/* Sort */}
          <select
            value={sortMode}
            onChange={(e) => setSortMode(e.target.value as SortMode)}
            className="px-6 py-3 bg-black border-2 border-white/20 text-white font-bold tracking-tight focus:outline-none focus:border-emerald-400 transition-colors"
          >
            <option value="newest">NEWEST FIRST</option>
            <option value="oldest">OLDEST FIRST</option>
            <option value="name-asc">NAME A-Z</option>
            <option value="name-desc">NAME Z-A</option>
            <option value="type">TYPE</option>
          </select>

          {/* Type filter */}
          {availableTypes.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {availableTypes.map(type => (
                <button
                  key={type}
                  onClick={() => toggleTypeFilter(type)}
                  className={`px-4 py-2 border-2 font-bold text-xs tracking-wide transition-all ${
                    selectedTypes.has(type)
                      ? 'bg-emerald-400 border-emerald-400 text-black'
                      : 'bg-transparent border-white/20 text-white/60 hover:border-white/40 hover:text-white'
                  }`}
                >
                  {type.toUpperCase()}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Bulk actions */}
        {selectedScaffolds.size > 0 && (
          <div className="mb-8 p-4 bg-red-500/10 border-2 border-red-500/50 flex items-center justify-between">
            <span className="font-mono text-sm text-red-400">
              {selectedScaffolds.size} SCAFFOLD{selectedScaffolds.size > 1 ? 'S' : ''} SELECTED
            </span>
            <div className="flex gap-3">
              <button
                onClick={() => setSelectedScaffolds(new Set())}
                className="px-4 py-2 bg-white/5 border-2 border-white/20 text-white font-bold text-sm hover:bg-white/10 transition-colors"
              >
                CLEAR SELECTION
              </button>
              <button
                onClick={handleBulkDelete}
                className="px-4 py-2 bg-red-500/20 border-2 border-red-500 text-red-400 font-bold text-sm hover:bg-red-500/30 transition-colors"
              >
                DELETE SELECTED
              </button>
            </div>
          </div>
        )}

        {/* Empty state */}
        {filteredScaffolds.length === 0 && (
          <div className="bg-white/[0.02] border-2 border-white/10 p-24 text-center">
            <div className="mb-8">
              <svg className="w-32 h-32 mx-auto text-white/10" fill="currentColor" viewBox="0 0 24 24">
                <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z" />
              </svg>
            </div>
            <h2 className="text-3xl font-black tracking-tight mb-4 text-white/60">
              {searchQuery || selectedTypes.size > 0 ? 'NO SCAFFOLDS FOUND' : 'NO SCAFFOLDS YET'}
            </h2>
            <p className="text-white/40 font-mono text-sm mb-8">
              {searchQuery || selectedTypes.size > 0
                ? 'TRY ADJUSTING YOUR FILTERS OR SEARCH QUERY'
                : 'CREATE YOUR FIRST SCAFFOLD TO GET STARTED'}
            </p>
            <button
              onClick={() => router.push('/')}
              className="px-8 py-4 bg-emerald-400 text-black font-bold tracking-tight hover:bg-emerald-300 transition-colors"
            >
              CREATE SCAFFOLD
            </button>
          </div>
        )}

        {/* Grid view */}
        {viewMode === 'grid' && filteredScaffolds.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredScaffolds.map((scaffold) => (
              <div
                key={scaffold.uuid}
                className="bg-white/[0.02] border-2 border-white/10 hover:border-emerald-400/50 transition-all duration-300 relative group overflow-hidden"
              >
                {/* Glow effect */}
                <div className="absolute inset-0 bg-gradient-to-br from-emerald-400/0 to-emerald-400/0 group-hover:from-emerald-400/5 group-hover:to-transparent transition-all duration-300 pointer-events-none" />

                {/* Selection checkbox */}
                <div className="absolute top-4 left-4 z-10">
                  <button
                    onClick={() => toggleScaffoldSelection(scaffold.uuid)}
                    className={`w-6 h-6 border-2 transition-all ${
                      selectedScaffolds.has(scaffold.uuid)
                        ? 'bg-emerald-400 border-emerald-400'
                        : 'bg-black/50 border-white/30 hover:border-white'
                    }`}
                  >
                    {selectedScaffolds.has(scaffold.uuid) && (
                      <svg className="w-full h-full text-black" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                      </svg>
                    )}
                  </button>
                </div>

                {/* Thumbnail */}
                <div className="aspect-video bg-gradient-to-br from-emerald-400/20 via-white/5 to-black relative overflow-hidden">
                  {scaffold.thumbnail_url ? (
                    <img src={scaffold.thumbnail_url} alt={scaffold.name} className="w-full h-full object-cover" />
                  ) : (
                    <div className="absolute inset-0 flex items-center justify-center">
                      <svg className="w-16 h-16 text-white/20" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z" />
                      </svg>
                    </div>
                  )}
                </div>

                {/* Content */}
                <div className="p-6">
                  <div className="mb-3">
                    <h3 className="text-xl font-black tracking-tight mb-2 truncate">{scaffold.name}</h3>
                    <div className="flex items-center gap-2">
                      <span className="px-2 py-1 bg-emerald-400/20 border border-emerald-400/50 text-emerald-400 text-xs font-mono tracking-wide">
                        {scaffold.type.toUpperCase()}
                      </span>
                      <span className="text-xs font-mono text-white/40">
                        {new Date(scaffold.created_at).toLocaleDateString()}
                      </span>
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="flex gap-2">
                    <button
                      onClick={() => handleLoad(scaffold.uuid)}
                      className="flex-1 px-4 py-2 bg-emerald-400 text-black font-bold text-sm tracking-tight hover:bg-emerald-300 transition-colors"
                    >
                      LOAD
                    </button>
                    <button
                      onClick={() => handleDuplicate(scaffold.uuid)}
                      className="px-4 py-2 bg-white/5 border-2 border-white/20 text-white font-bold text-sm hover:border-emerald-400 transition-colors"
                      title="Duplicate"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                      </svg>
                    </button>
                    <button
                      onClick={() => handleDelete(scaffold.uuid)}
                      className="px-4 py-2 bg-red-500/20 border-2 border-red-500/50 text-red-400 font-bold text-sm hover:bg-red-500/30 transition-colors"
                      title="Delete"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                      </svg>
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* List view */}
        {viewMode === 'list' && filteredScaffolds.length > 0 && (
          <div className="space-y-2">
            {filteredScaffolds.map((scaffold) => (
              <div
                key={scaffold.uuid}
                className="bg-white/[0.02] border-2 border-white/10 hover:border-emerald-400/50 transition-all duration-300 p-6 flex items-center gap-6 relative group"
              >
                {/* Glow effect */}
                <div className="absolute inset-0 bg-gradient-to-r from-emerald-400/0 to-emerald-400/0 group-hover:from-emerald-400/5 group-hover:to-transparent transition-all duration-300 pointer-events-none" />

                {/* Selection */}
                <div className="relative z-10">
                  <button
                    onClick={() => toggleScaffoldSelection(scaffold.uuid)}
                    className={`w-6 h-6 border-2 transition-all ${
                      selectedScaffolds.has(scaffold.uuid)
                        ? 'bg-emerald-400 border-emerald-400'
                        : 'bg-black border-white/30 hover:border-white'
                    }`}
                  >
                    {selectedScaffolds.has(scaffold.uuid) && (
                      <svg className="w-full h-full text-black" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                      </svg>
                    )}
                  </button>
                </div>

                {/* Thumbnail */}
                <div className="w-24 h-16 bg-gradient-to-br from-emerald-400/20 via-white/5 to-black flex items-center justify-center shrink-0">
                  {scaffold.thumbnail_url ? (
                    <img src={scaffold.thumbnail_url} alt={scaffold.name} className="w-full h-full object-cover" />
                  ) : (
                    <svg className="w-8 h-8 text-white/20" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z" />
                    </svg>
                  )}
                </div>

                {/* Info */}
                <div className="flex-1 min-w-0">
                  <h3 className="text-lg font-black tracking-tight mb-1 truncate">{scaffold.name}</h3>
                  <div className="flex items-center gap-4 text-xs font-mono text-white/40">
                    <span className="px-2 py-1 bg-emerald-400/20 border border-emerald-400/50 text-emerald-400">
                      {scaffold.type.toUpperCase()}
                    </span>
                    <span>{new Date(scaffold.created_at).toLocaleDateString()}</span>
                  </div>
                </div>

                {/* Actions */}
                <div className="flex gap-2 shrink-0">
                  <button
                    onClick={() => handleLoad(scaffold.uuid)}
                    className="px-6 py-2 bg-emerald-400 text-black font-bold text-sm tracking-tight hover:bg-emerald-300 transition-colors"
                  >
                    LOAD
                  </button>
                  <button
                    onClick={() => handleDuplicate(scaffold.uuid)}
                    className="px-4 py-2 bg-white/5 border-2 border-white/20 text-white font-bold text-sm hover:border-emerald-400 transition-colors"
                  >
                    DUPLICATE
                  </button>
                  <button
                    onClick={() => handleDelete(scaffold.uuid)}
                    className="px-4 py-2 bg-red-500/20 border-2 border-red-500/50 text-red-400 font-bold text-sm hover:bg-red-500/30 transition-colors"
                  >
                    DELETE
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Delete Confirmation Modal */}
      {showDeleteModal && (
        <div className="fixed inset-0 bg-black/90 backdrop-blur-sm flex items-center justify-center z-50 p-8 animate-in fade-in duration-200">
          <div className="bg-black border-4 border-red-500 max-w-lg w-full p-8 relative">
            <div className="absolute top-0 right-0 w-48 h-48 bg-gradient-radial from-red-500/20 to-transparent pointer-events-none" />

            <div className="relative">
              <h2 className="text-3xl font-black tracking-tighter text-red-400 mb-6">
                DELETE SCAFFOLD?
              </h2>

              <p className="text-white/60 font-mono text-sm mb-8">
                THIS ACTION CANNOT BE UNDONE. THE SCAFFOLD WILL BE PERMANENTLY DELETED.
              </p>

              <div className="flex gap-4">
                <button
                  onClick={() => {
                    setShowDeleteModal(false);
                    setScaffoldToDelete(null);
                  }}
                  className="flex-1 px-6 py-4 bg-white/5 border-2 border-white/20 text-white font-bold tracking-tight hover:bg-white/10 transition-colors"
                >
                  CANCEL
                </button>
                <button
                  onClick={confirmDelete}
                  className="flex-1 px-6 py-4 bg-red-500 border-2 border-red-500 text-white font-bold tracking-tight hover:bg-red-600 transition-colors"
                >
                  DELETE
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
