'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { LogoIcon } from '@/components/Logo';
import { useAuthStore } from '@/lib/store/authStore';
import { Settings, User, LogOut, Menu, X } from 'lucide-react';

type PageType = 'dashboard' | 'generator' | 'library' | 'settings' | 'account';

interface NavHeaderProps {
  currentPage: PageType;
}

export function NavHeader({ currentPage }: NavHeaderProps) {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [userDropdownOpen, setUserDropdownOpen] = useState(false);
  const router = useRouter();
  const { user, logout, isAuthenticated } = useAuthStore();

  // Close dropdowns on escape key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setUserDropdownOpen(false);
        setMobileMenuOpen(false);
      }
    };
    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, []);

  // Close dropdowns when clicking outside
  useEffect(() => {
    const handleClickOutside = () => {
      setUserDropdownOpen(false);
    };
    if (userDropdownOpen) {
      document.addEventListener('click', handleClickOutside);
      return () => document.removeEventListener('click', handleClickOutside);
    }
  }, [userDropdownOpen]);

  const handleLogout = () => {
    logout();
    router.push('/login');
  };

  const navigate = (path: string) => {
    router.push(path);
    setMobileMenuOpen(false);
  };

  const navLinks = [
    { label: 'DASHBOARD', path: '/dashboard', id: 'dashboard' as PageType },
    { label: 'GENERATOR', path: '/generator', id: 'generator' as PageType },
    { label: 'LIBRARY', path: '/library', id: 'library' as PageType },
  ];

  return (
    <header className="sticky top-0 z-50 w-full">
      {/* Background with gradient border */}
      <div className="relative">
        {/* Top gradient border */}
        <div className="absolute inset-x-0 top-0 h-[2px] bg-gradient-to-r from-transparent via-emerald-500 to-transparent opacity-60" />

        {/* Main header */}
        <div className="bg-black/90 backdrop-blur-xl border-b border-emerald-500/20">
          <nav className="container mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex h-16 items-center justify-between">
              {/* Logo */}
              <div
                className="flex items-center gap-3 cursor-pointer group"
                onClick={() => navigate('/dashboard')}
              >
                <div className="relative">
                  <div className="absolute inset-0 bg-emerald-500/20 blur-xl opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
                  <LogoIcon size={36} className="relative z-10 transition-transform duration-300 group-hover:scale-110" />
                </div>
                <div className="hidden sm:flex flex-col">
                  <span className="text-lg font-black tracking-tight text-white">
                    MorphoStruct
                  </span>
                  <span className="text-[9px] font-mono text-emerald-400/70 tracking-[0.15em] uppercase">
                    Bioprinting Platform
                  </span>
                </div>
              </div>

              {/* Desktop Navigation */}
              <div className="hidden md:flex items-center gap-1">
                {navLinks.map((link) => (
                  <button
                    key={link.id}
                    onClick={() => navigate(link.path)}
                    className={`
                      relative px-4 py-2 font-mono text-sm tracking-wider transition-all duration-300
                      ${currentPage === link.id
                        ? 'text-emerald-400'
                        : 'text-gray-400 hover:text-emerald-300'
                      }
                    `}
                  >
                    {/* Hover background glow */}
                    <div className={`
                      absolute inset-0 bg-emerald-500/10 opacity-0 hover:opacity-100 transition-opacity duration-300
                      ${currentPage === link.id ? 'opacity-100 bg-emerald-500/15' : ''}
                    `} />

                    {/* Text */}
                    <span className="relative z-10">{link.label}</span>

                    {/* Active underline */}
                    {currentPage === link.id && (
                      <div className="absolute bottom-0 left-0 right-0 h-[2px] bg-gradient-to-r from-transparent via-emerald-500 to-transparent" />
                    )}
                  </button>
                ))}
              </div>

              {/* Right side - Desktop */}
              <div className="hidden md:flex items-center gap-2">
                {/* Settings button */}
                <button
                  onClick={() => navigate('/settings')}
                  className={`
                    relative p-2 rounded-sm transition-all duration-300 group
                    ${currentPage === 'settings' ? 'text-emerald-400' : 'text-gray-400 hover:text-emerald-300'}
                  `}
                  aria-label="Settings"
                >
                  <div className="absolute inset-0 bg-emerald-500/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                  <Settings className="w-5 h-5 relative z-10" />
                </button>

                {/* User dropdown */}
                {isAuthenticated && user && (
                  <div className="relative">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setUserDropdownOpen(!userDropdownOpen);
                      }}
                      className={`
                        relative flex items-center gap-2 px-3 py-2 rounded-sm transition-all duration-300 group
                        ${currentPage === 'account' ? 'text-emerald-400' : 'text-gray-400 hover:text-emerald-300'}
                      `}
                    >
                      <div className="absolute inset-0 bg-emerald-500/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                      <User className="w-5 h-5 relative z-10" />
                      <span className="font-mono text-sm relative z-10">{user.username}</span>
                    </button>

                    {/* Dropdown menu */}
                    {userDropdownOpen && (
                      <div className="absolute right-0 mt-2 w-48 origin-top-right">
                        {/* Glow effect */}
                        <div className="absolute inset-0 bg-emerald-500/20 blur-xl" />

                        {/* Menu */}
                        <div className="relative bg-black/95 backdrop-blur-xl border border-emerald-500/30 shadow-2xl overflow-hidden">
                          {/* Top accent line */}
                          <div className="h-[1px] bg-gradient-to-r from-transparent via-emerald-500 to-transparent" />

                          <div className="py-1">
                            <div className="px-4 py-2 border-b border-emerald-500/20">
                              <p className="text-xs font-mono text-emerald-400/70 tracking-wider uppercase">
                                User Menu
                              </p>
                            </div>

                            <button
                              onClick={() => {
                                navigate('/account');
                                setUserDropdownOpen(false);
                              }}
                              className="w-full text-left px-4 py-2 text-sm text-gray-300 hover:bg-emerald-500/10 hover:text-emerald-400 transition-all duration-200 font-mono"
                            >
                              Account
                            </button>

                            <button
                              onClick={() => {
                                navigate('/settings');
                                setUserDropdownOpen(false);
                              }}
                              className="w-full text-left px-4 py-2 text-sm text-gray-300 hover:bg-emerald-500/10 hover:text-emerald-400 transition-all duration-200 font-mono"
                            >
                              Settings
                            </button>

                            <div className="h-[1px] bg-gradient-to-r from-transparent via-emerald-500/30 to-transparent my-1" />

                            <button
                              onClick={handleLogout}
                              className="w-full text-left px-4 py-2 text-sm text-red-400 hover:bg-red-500/10 hover:text-red-300 transition-all duration-200 font-mono flex items-center gap-2"
                            >
                              <LogOut className="w-4 h-4" />
                              Logout
                            </button>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Mobile menu button */}
              <button
                onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                className="md:hidden p-2 text-gray-400 hover:text-emerald-300 transition-colors duration-300"
                aria-label="Toggle menu"
              >
                {mobileMenuOpen ? (
                  <X className="w-6 h-6" />
                ) : (
                  <Menu className="w-6 h-6" />
                )}
              </button>
            </div>
          </nav>

          {/* Mobile menu */}
          {mobileMenuOpen && (
            <div className="md:hidden border-t border-emerald-500/20">
              <div className="px-4 py-3 space-y-1 bg-black/50">
                {navLinks.map((link) => (
                  <button
                    key={link.id}
                    onClick={() => navigate(link.path)}
                    className={`
                      block w-full text-left px-4 py-3 font-mono text-sm tracking-wider transition-all duration-200
                      ${currentPage === link.id
                        ? 'text-emerald-400 bg-emerald-500/15 border-l-2 border-emerald-500'
                        : 'text-gray-400 hover:text-emerald-300 hover:bg-emerald-500/10 border-l-2 border-transparent'
                      }
                    `}
                  >
                    {link.label}
                  </button>
                ))}

                <div className="h-[1px] bg-gradient-to-r from-transparent via-emerald-500/30 to-transparent my-2" />

                <button
                  onClick={() => navigate('/settings')}
                  className={`
                    block w-full text-left px-4 py-3 font-mono text-sm tracking-wider transition-all duration-200
                    ${currentPage === 'settings'
                      ? 'text-emerald-400 bg-emerald-500/15 border-l-2 border-emerald-500'
                      : 'text-gray-400 hover:text-emerald-300 hover:bg-emerald-500/10 border-l-2 border-transparent'
                    }
                  `}
                >
                  SETTINGS
                </button>

                {isAuthenticated && user && (
                  <>
                    <button
                      onClick={() => navigate('/account')}
                      className={`
                        block w-full text-left px-4 py-3 font-mono text-sm tracking-wider transition-all duration-200
                        ${currentPage === 'account'
                          ? 'text-emerald-400 bg-emerald-500/15 border-l-2 border-emerald-500'
                          : 'text-gray-400 hover:text-emerald-300 hover:bg-emerald-500/10 border-l-2 border-transparent'
                        }
                      `}
                    >
                      ACCOUNT ({user.username})
                    </button>

                    <button
                      onClick={handleLogout}
                      className="block w-full text-left px-4 py-3 font-mono text-sm tracking-wider text-red-400 hover:text-red-300 hover:bg-red-500/10 transition-all duration-200 border-l-2 border-transparent"
                    >
                      LOGOUT
                    </button>
                  </>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Bottom gradient accent */}
        <div className="absolute inset-x-0 bottom-0 h-[1px] bg-gradient-to-r from-transparent via-emerald-500/40 to-transparent" />
      </div>
    </header>
  );
}
