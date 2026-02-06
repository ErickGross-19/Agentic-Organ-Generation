'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useAuthStore } from '@/lib/store/authStore';
import {
  getUserProfile,
  updateProfile,
  sendVerificationEmail,
  changePassword,
  getSessions,
  revokeSession,
  revokeAllSessions,
  deleteAccount,
  UserProfile,
  Session
} from '@/lib/api/auth';
import { NavHeader } from '@/components/NavHeader';

type TabType = 'profile' | 'security' | 'sessions' | 'danger';

export default function AccountPage() {
  const router = useRouter();
  const { user, logout } = useAuthStore();
  const [activeTab, setActiveTab] = useState<TabType>('profile');
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  // Profile state
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [displayName, setDisplayName] = useState('');
  const [email, setEmail] = useState('');
  const [avatarUrl, setAvatarUrl] = useState('');

  // Password state
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showCurrentPassword, setShowCurrentPassword] = useState(false);
  const [showNewPassword, setShowNewPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);

  // Sessions state
  const [sessions, setSessions] = useState<Session[]>([]);

  // Delete modal state
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [deletePassword, setDeletePassword] = useState('');
  const [deleteConfirm, setDeleteConfirm] = useState('');

  useEffect(() => {
    loadProfile();
  }, []);

  useEffect(() => {
    if (activeTab === 'sessions') {
      loadSessions();
    }
  }, [activeTab]);

  const loadProfile = async () => {
    try {
      const data = await getUserProfile();
      setProfile(data);
      setDisplayName(data.display_name || '');
      setEmail(data.email || '');
      setAvatarUrl(data.avatar_url || '');
    } catch (err) {
      console.error('Failed to load profile:', err);
      setError('FAILED TO LOAD PROFILE');
    } finally {
      setLoading(false);
    }
  };

  const loadSessions = async () => {
    try {
      const data = await getSessions();
      setSessions(data);
    } catch (err) {
      console.error('Failed to load sessions:', err);
      setError('FAILED TO LOAD SESSIONS');
    }
  };

  const showMessage = (msg: string, isError = false) => {
    if (isError) {
      setError(msg);
      setMessage('');
    } else {
      setMessage(msg);
      setError('');
    }
    setTimeout(() => {
      setMessage('');
      setError('');
    }, 3000);
  };

  const handleSaveProfile = async () => {
    setSaving(true);
    try {
      const updated = await updateProfile({
        display_name: displayName || null,
        email: email || null,
        avatar_url: avatarUrl || null,
      });
      setProfile(updated);
      showMessage('PROFILE UPDATED SUCCESSFULLY');
    } catch (err) {
      showMessage('PROFILE UPDATE FAILED', true);
    } finally {
      setSaving(false);
    }
  };

  const handleSendVerification = async () => {
    setSaving(true);
    try {
      await sendVerificationEmail();
      showMessage('VERIFICATION EMAIL SENT');
    } catch (err) {
      showMessage('FAILED TO SEND VERIFICATION EMAIL', true);
    } finally {
      setSaving(false);
    }
  };

  const handleChangePassword = async () => {
    if (newPassword !== confirmPassword) {
      showMessage('PASSWORDS DO NOT MATCH', true);
      return;
    }

    if (newPassword.length < 8) {
      showMessage('PASSWORD MUST BE AT LEAST 8 CHARACTERS', true);
      return;
    }

    setSaving(true);
    try {
      await changePassword(currentPassword, newPassword);
      setCurrentPassword('');
      setNewPassword('');
      setConfirmPassword('');
      showMessage('PASSWORD CHANGED SUCCESSFULLY');
    } catch (err) {
      showMessage(err instanceof Error ? err.message.toUpperCase() : 'PASSWORD CHANGE FAILED', true);
    } finally {
      setSaving(false);
    }
  };

  const handleRevokeSession = async (sessionId: number) => {
    try {
      await revokeSession(sessionId);
      showMessage('SESSION REVOKED');
      loadSessions();
    } catch (err) {
      showMessage('FAILED TO REVOKE SESSION', true);
    }
  };

  const handleRevokeAllSessions = async () => {
    if (!confirm('REVOKE ALL OTHER SESSIONS? THIS WILL LOG OUT ALL OTHER DEVICES.')) return;

    try {
      await revokeAllSessions();
      showMessage('ALL OTHER SESSIONS REVOKED');
      loadSessions();
    } catch (err) {
      showMessage('FAILED TO REVOKE SESSIONS', true);
    }
  };

  const handleDeleteAccount = async () => {
    if (deleteConfirm !== 'DELETE') {
      showMessage('TYPE "DELETE" TO CONFIRM', true);
      return;
    }

    if (!deletePassword) {
      showMessage('PASSWORD REQUIRED', true);
      return;
    }

    setSaving(true);
    try {
      await deleteAccount(deletePassword);
      showMessage('ACCOUNT DELETED');
      setTimeout(() => {
        logout();
        router.push('/login');
      }, 1500);
    } catch (err) {
      showMessage(err instanceof Error ? err.message.toUpperCase() : 'ACCOUNT DELETION FAILED', true);
      setSaving(false);
    }
  };

  const passwordRequirements = [
    { label: 'At least 8 characters', met: newPassword.length >= 8 },
    { label: 'Passwords match', met: newPassword.length > 0 && newPassword === confirmPassword },
  ];

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
      <NavHeader currentPage="account" />

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

      <div className="relative max-w-6xl mx-auto px-8 py-12">
        {/* Page Title */}
        <div className="mb-12 pb-8 border-b-4 border-white/10">
          <div className="text-xs font-mono text-emerald-400 tracking-widest mb-4">ACCOUNT_MANAGEMENT</div>
          <h1 className="text-6xl font-black tracking-tighter mb-4" style={{ fontFamily: 'system-ui' }}>
            ACCOUNT
          </h1>
          <div className="flex items-center gap-4 text-sm font-mono">
            <span className="text-white/40">USER:</span>
            <span className="text-white font-bold">{user?.username}</span>
            <span className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse" />
            <span className="text-emerald-400">AUTHENTICATED</span>
          </div>
        </div>

        {/* Status messages */}
        {message && (
          <div className="mb-8 p-4 bg-emerald-400/10 border-l-4 border-emerald-400 font-mono text-sm tracking-wide animate-pulse">
            <span className="text-emerald-400">▶</span> {message}
          </div>
        )}
        {error && (
          <div className="mb-8 p-4 bg-red-500/10 border-l-4 border-red-500 font-mono text-sm tracking-wide animate-pulse">
            <span className="text-red-400">▶</span> {error}
          </div>
        )}

        {/* Tab Navigation */}
        <nav className="mb-12 flex gap-1 border-b border-white/10">
          {[
            { id: 'profile', label: 'PROFILE' },
            { id: 'security', label: 'SECURITY' },
            { id: 'sessions', label: 'SESSIONS' },
            { id: 'danger', label: 'DANGER ZONE' }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as TabType)}
              className={`px-8 py-4 font-bold tracking-tight transition-all relative ${
                activeTab === tab.id
                  ? tab.id === 'danger'
                    ? 'bg-red-500/20 text-red-400 border-b-4 border-red-500'
                    : 'bg-white text-black'
                  : tab.id === 'danger'
                    ? 'text-red-400/40 hover:text-red-400 hover:bg-red-500/5'
                    : 'text-white/40 hover:text-white hover:bg-white/5'
              }`}
            >
              {tab.label}
              {activeTab === tab.id && tab.id !== 'danger' && (
                <div className="absolute bottom-0 left-0 right-0 h-1 bg-emerald-400" />
              )}
            </button>
          ))}
        </nav>

        {/* Content Sections */}
        <div className="space-y-8">
          {/* PROFILE SECTION */}
          {activeTab === 'profile' && (
            <div className="space-y-8 animate-in fade-in duration-300">
              <div className="border-l-4 border-white/20 pl-6">
                <p className="font-mono text-sm text-white/60 leading-relaxed">
                  MANAGE YOUR PERSONAL INFORMATION AND ACCOUNT DETAILS.
                </p>
              </div>

              <div className="bg-white/[0.02] border-2 border-white/10 p-8 hover:border-emerald-400/30 transition-all duration-300 relative group">
                <div className="absolute top-0 right-0 w-32 h-32 bg-gradient-radial from-emerald-400/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />

                <h3 className="text-2xl font-black tracking-tight mb-6">PROFILE INFORMATION</h3>

                <div className="space-y-6">
                  {/* Display Name */}
                  <div>
                    <label className="text-sm font-mono text-white/60 mb-2 block tracking-wide">
                      DISPLAY NAME
                    </label>
                    <input
                      type="text"
                      value={displayName}
                      onChange={(e) => setDisplayName(e.target.value)}
                      placeholder="How you want to be referred to"
                      className="w-full px-4 py-3 bg-black border-2 border-white/20 text-white placeholder-white/20 focus:outline-none focus:border-emerald-400 font-mono text-sm transition-colors"
                    />
                  </div>

                  {/* Email */}
                  <div>
                    <label className="text-sm font-mono text-white/60 mb-2 block tracking-wide">
                      EMAIL ADDRESS
                    </label>
                    <div className="flex gap-2">
                      <input
                        type="email"
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        placeholder="your@email.com"
                        className="flex-1 px-4 py-3 bg-black border-2 border-white/20 text-white placeholder-white/20 focus:outline-none focus:border-emerald-400 font-mono text-sm transition-colors"
                      />
                      {profile?.email && (
                        <div className={`flex items-center gap-2 px-4 py-3 border-2 ${
                          profile.email_verified
                            ? 'bg-emerald-400/20 border-emerald-400/50'
                            : 'bg-yellow-500/20 border-yellow-500/50'
                        }`}>
                          <div className={`w-2 h-2 rounded-full ${
                            profile.email_verified ? 'bg-emerald-400' : 'bg-yellow-500'
                          }`} />
                          <span className={`text-xs font-mono tracking-wide ${
                            profile.email_verified ? 'text-emerald-400' : 'text-yellow-500'
                          }`}>
                            {profile.email_verified ? 'VERIFIED' : 'UNVERIFIED'}
                          </span>
                        </div>
                      )}
                    </div>
                    {profile?.email && !profile.email_verified && (
                      <button
                        onClick={handleSendVerification}
                        disabled={saving}
                        className="mt-2 px-4 py-2 bg-yellow-500/20 border border-yellow-500/50 text-yellow-400 text-xs font-mono tracking-wide hover:bg-yellow-500/30 transition-colors disabled:opacity-50"
                      >
                        SEND VERIFICATION EMAIL
                      </button>
                    )}
                  </div>

                  {/* Avatar URL */}
                  <div>
                    <label className="text-sm font-mono text-white/60 mb-2 block tracking-wide">
                      AVATAR URL
                    </label>
                    <div className="flex gap-4">
                      <input
                        type="url"
                        value={avatarUrl}
                        onChange={(e) => setAvatarUrl(e.target.value)}
                        placeholder="https://..."
                        className="flex-1 px-4 py-3 bg-black border-2 border-white/20 text-white placeholder-white/20 focus:outline-none focus:border-emerald-400 font-mono text-sm transition-colors"
                      />
                      {avatarUrl && (
                        <div className="w-12 h-12 border-2 border-white/20 overflow-hidden flex items-center justify-center bg-white/5">
                          <img src={avatarUrl} alt="Avatar preview" className="w-full h-full object-cover" onError={(e) => {
                            e.currentTarget.style.display = 'none';
                          }} />
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Save Button */}
                  <div className="pt-4">
                    <button
                      onClick={handleSaveProfile}
                      disabled={saving}
                      className="w-full px-6 py-4 bg-emerald-400 text-black font-bold tracking-tight hover:bg-emerald-300 disabled:bg-white/10 disabled:text-white/30 transition-colors relative overflow-hidden group"
                    >
                      <span className="relative z-10">
                        {saving ? 'SAVING...' : 'SAVE PROFILE'}
                      </span>
                    </button>
                  </div>
                </div>
              </div>

              {/* Account Info */}
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-white/[0.02] border border-white/10 p-4">
                  <div className="text-xs font-mono text-white/40 mb-2 tracking-wide">USERNAME</div>
                  <div className="text-xl font-black tracking-tight">{profile?.username}</div>
                </div>
                <div className="bg-white/[0.02] border border-white/10 p-4">
                  <div className="text-xs font-mono text-white/40 mb-2 tracking-wide">MEMBER SINCE</div>
                  <div className="text-xl font-black tracking-tight">
                    {profile?.created_at ? new Date(profile.created_at).toLocaleDateString() : '-'}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* SECURITY SECTION */}
          {activeTab === 'security' && (
            <div className="space-y-8 animate-in fade-in duration-300">
              <div className="border-l-4 border-white/20 pl-6">
                <p className="font-mono text-sm text-white/60 leading-relaxed">
                  SECURE YOUR ACCOUNT WITH A STRONG PASSWORD.
                </p>
              </div>

              <div className="bg-white/[0.02] border-2 border-white/10 p-8 hover:border-emerald-400/30 transition-all duration-300">
                <h3 className="text-2xl font-black tracking-tight mb-6">CHANGE PASSWORD</h3>

                <div className="space-y-6">
                  {/* Current Password */}
                  <div>
                    <label className="text-sm font-mono text-white/60 mb-2 block tracking-wide">
                      CURRENT PASSWORD
                    </label>
                    <div className="flex gap-2">
                      <input
                        type={showCurrentPassword ? "text" : "password"}
                        value={currentPassword}
                        onChange={(e) => setCurrentPassword(e.target.value)}
                        placeholder="Enter current password"
                        className="flex-1 px-4 py-3 bg-black border-2 border-white/20 text-white placeholder-white/20 focus:outline-none focus:border-emerald-400 font-mono text-sm transition-colors"
                      />
                      <button
                        onClick={() => setShowCurrentPassword(!showCurrentPassword)}
                        className="px-4 py-3 bg-white/5 border-2 border-white/20 hover:border-white/40 transition-colors font-mono text-xs"
                      >
                        {showCurrentPassword ? 'HIDE' : 'SHOW'}
                      </button>
                    </div>
                  </div>

                  {/* New Password */}
                  <div>
                    <label className="text-sm font-mono text-white/60 mb-2 block tracking-wide">
                      NEW PASSWORD
                    </label>
                    <div className="flex gap-2">
                      <input
                        type={showNewPassword ? "text" : "password"}
                        value={newPassword}
                        onChange={(e) => setNewPassword(e.target.value)}
                        placeholder="Enter new password"
                        className="flex-1 px-4 py-3 bg-black border-2 border-white/20 text-white placeholder-white/20 focus:outline-none focus:border-emerald-400 font-mono text-sm transition-colors"
                      />
                      <button
                        onClick={() => setShowNewPassword(!showNewPassword)}
                        className="px-4 py-3 bg-white/5 border-2 border-white/20 hover:border-white/40 transition-colors font-mono text-xs"
                      >
                        {showNewPassword ? 'HIDE' : 'SHOW'}
                      </button>
                    </div>
                  </div>

                  {/* Confirm Password */}
                  <div>
                    <label className="text-sm font-mono text-white/60 mb-2 block tracking-wide">
                      CONFIRM NEW PASSWORD
                    </label>
                    <div className="flex gap-2">
                      <input
                        type={showConfirmPassword ? "text" : "password"}
                        value={confirmPassword}
                        onChange={(e) => setConfirmPassword(e.target.value)}
                        placeholder="Confirm new password"
                        className="flex-1 px-4 py-3 bg-black border-2 border-white/20 text-white placeholder-white/20 focus:outline-none focus:border-emerald-400 font-mono text-sm transition-colors"
                      />
                      <button
                        onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                        className="px-4 py-3 bg-white/5 border-2 border-white/20 hover:border-white/40 transition-colors font-mono text-xs"
                      >
                        {showConfirmPassword ? 'HIDE' : 'SHOW'}
                      </button>
                    </div>
                  </div>

                  {/* Password Requirements */}
                  {newPassword && (
                    <div className="bg-black/50 border-l-4 border-emerald-400/50 p-4">
                      <div className="text-xs font-mono text-white/60 mb-3 tracking-wide">PASSWORD REQUIREMENTS:</div>
                      <div className="space-y-2">
                        {passwordRequirements.map((req, i) => (
                          <div key={i} className="flex items-center gap-2">
                            <div className={`w-3 h-3 border-2 ${
                              req.met ? 'bg-emerald-400 border-emerald-400' : 'border-white/20'
                            }`} />
                            <span className={`text-xs font-mono ${
                              req.met ? 'text-emerald-400' : 'text-white/40'
                            }`}>
                              {req.label}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Change Password Button */}
                  <div className="pt-4">
                    <button
                      onClick={handleChangePassword}
                      disabled={!currentPassword || !newPassword || !confirmPassword || saving}
                      className="w-full px-6 py-4 bg-emerald-400 text-black font-bold tracking-tight hover:bg-emerald-300 disabled:bg-white/10 disabled:text-white/30 transition-colors"
                    >
                      {saving ? 'CHANGING PASSWORD...' : 'CHANGE PASSWORD'}
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* SESSIONS SECTION */}
          {activeTab === 'sessions' && (
            <div className="space-y-8 animate-in fade-in duration-300">
              <div className="border-l-4 border-white/20 pl-6">
                <p className="font-mono text-sm text-white/60 leading-relaxed">
                  MANAGE ACTIVE SESSIONS ACROSS ALL YOUR DEVICES.
                </p>
              </div>

              <div className="space-y-4">
                {sessions.map((session) => (
                  <div
                    key={session.id}
                    className={`bg-white/[0.02] border-2 p-6 transition-all duration-300 ${
                      session.is_current
                        ? 'border-emerald-400/50 bg-emerald-400/5'
                        : 'border-white/10 hover:border-white/20'
                    }`}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-3 mb-3">
                          <h4 className="text-lg font-black tracking-tight">{session.user_agent}</h4>
                          {session.is_current && (
                            <div className="flex items-center gap-2 px-3 py-1 bg-emerald-400/20 border border-emerald-400/50">
                              <div className="w-1.5 h-1.5 bg-emerald-400 rounded-full animate-pulse" />
                              <span className="text-xs font-mono text-emerald-400 tracking-wide">CURRENT SESSION</span>
                            </div>
                          )}
                        </div>
                        <div className="space-y-1 text-sm font-mono text-white/40">
                          <div>IP: <span className="text-white/60">{session.ip_address}</span></div>
                          <div>LAST ACTIVE: <span className="text-white/60">
                            {new Date(session.last_active).toLocaleString()}
                          </span></div>
                        </div>
                      </div>
                      {!session.is_current && (
                        <button
                          onClick={() => handleRevokeSession(session.id)}
                          className="px-4 py-2 bg-red-500/20 border border-red-500/50 text-red-400 font-bold text-sm tracking-tight hover:bg-red-500/30 transition-colors"
                        >
                          REVOKE
                        </button>
                      )}
                    </div>
                  </div>
                ))}
              </div>

              {sessions.length > 1 && (
                <button
                  onClick={handleRevokeAllSessions}
                  className="w-full px-6 py-4 bg-red-500/20 border-2 border-red-500/50 text-red-400 font-bold tracking-tight hover:bg-red-500/30 hover:border-red-500 transition-all"
                >
                  REVOKE ALL OTHER SESSIONS
                </button>
              )}

              {sessions.length === 0 && (
                <div className="bg-white/[0.02] border border-white/10 p-12 text-center">
                  <div className="text-white/40 font-mono text-sm">NO ACTIVE SESSIONS</div>
                </div>
              )}
            </div>
          )}

          {/* DANGER ZONE SECTION */}
          {activeTab === 'danger' && (
            <div className="space-y-8 animate-in fade-in duration-300">
              <div className="border-l-4 border-red-500/50 pl-6">
                <p className="font-mono text-sm text-red-400/60 leading-relaxed">
                  IRREVERSIBLE ACTIONS. PROCEED WITH EXTREME CAUTION.
                </p>
              </div>

              <div className="bg-red-500/5 border-2 border-red-500/30 p-8">
                <h3 className="text-2xl font-black tracking-tight text-red-400 mb-6">DELETE ACCOUNT</h3>

                <div className="space-y-6">
                  <div className="bg-black/50 border-l-4 border-red-500 p-4">
                    <div className="text-xs font-mono text-red-400/80 tracking-wide space-y-2">
                      <div>⚠ THIS ACTION CANNOT BE UNDONE</div>
                      <div>⚠ ALL DATA WILL BE PERMANENTLY DELETED</div>
                      <div>⚠ ALL ACTIVE SESSIONS WILL BE TERMINATED</div>
                      <div>⚠ THIS INCLUDES ALL SCAFFOLDS AND PROJECTS</div>
                    </div>
                  </div>

                  <button
                    onClick={() => setShowDeleteModal(true)}
                    className="w-full px-6 py-4 bg-red-500/20 border-2 border-red-500 text-red-400 font-bold tracking-tight hover:bg-red-500/40 transition-all relative group overflow-hidden"
                  >
                    <span className="relative z-10">DELETE ACCOUNT</span>
                    <div className="absolute inset-0 bg-red-500/20 translate-y-full group-hover:translate-y-0 transition-transform duration-300" />
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Delete Confirmation Modal */}
      {showDeleteModal && (
        <div className="fixed inset-0 bg-black/90 backdrop-blur-sm flex items-center justify-center z-50 p-8 animate-in fade-in duration-200">
          <div className="bg-black border-4 border-red-500 max-w-2xl w-full p-8 relative">
            <div className="absolute top-0 right-0 w-64 h-64 bg-gradient-radial from-red-500/20 to-transparent pointer-events-none" />

            <div className="relative">
              <h2 className="text-4xl font-black tracking-tighter text-red-400 mb-6">
                CONFIRM ACCOUNT DELETION
              </h2>

              <div className="space-y-6 mb-8">
                <div className="bg-red-500/10 border-l-4 border-red-500 p-4">
                  <div className="text-sm font-mono text-red-400/80 space-y-2">
                    <div>THIS WILL PERMANENTLY DELETE:</div>
                    <div className="pl-4">• Your account and profile</div>
                    <div className="pl-4">• All scaffolds and projects</div>
                    <div className="pl-4">• All API keys and preferences</div>
                    <div className="pl-4">• All session data</div>
                  </div>
                </div>

                <div>
                  <label className="text-sm font-mono text-red-400/60 mb-2 block tracking-wide">
                    ENTER YOUR PASSWORD
                  </label>
                  <input
                    type="password"
                    value={deletePassword}
                    onChange={(e) => setDeletePassword(e.target.value)}
                    placeholder="Password"
                    className="w-full px-4 py-3 bg-black border-2 border-red-500/50 text-white placeholder-red-400/20 focus:outline-none focus:border-red-500 font-mono text-sm transition-colors"
                  />
                </div>

                <div>
                  <label className="text-sm font-mono text-red-400/60 mb-2 block tracking-wide">
                    TYPE "DELETE" TO CONFIRM
                  </label>
                  <input
                    type="text"
                    value={deleteConfirm}
                    onChange={(e) => setDeleteConfirm(e.target.value)}
                    placeholder="DELETE"
                    className="w-full px-4 py-3 bg-black border-2 border-red-500/50 text-white placeholder-red-400/20 focus:outline-none focus:border-red-500 font-mono text-sm transition-colors uppercase"
                  />
                </div>
              </div>

              <div className="flex gap-4">
                <button
                  onClick={() => {
                    setShowDeleteModal(false);
                    setDeletePassword('');
                    setDeleteConfirm('');
                  }}
                  className="flex-1 px-6 py-4 bg-white/5 border-2 border-white/20 text-white font-bold tracking-tight hover:bg-white/10 transition-colors"
                >
                  CANCEL
                </button>
                <button
                  onClick={handleDeleteAccount}
                  disabled={deleteConfirm !== 'DELETE' || !deletePassword || saving}
                  className="flex-1 px-6 py-4 bg-red-500 border-2 border-red-500 text-white font-bold tracking-tight hover:bg-red-600 disabled:bg-red-500/20 disabled:border-red-500/50 disabled:text-red-400/50 transition-colors"
                >
                  {saving ? 'DELETING...' : 'DELETE ACCOUNT'}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
