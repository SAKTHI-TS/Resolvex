import { Link, useNavigate } from 'react-router-dom';
import { Shield, Bell, Menu, X, LogOut, User } from 'lucide-react';
import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { LanguageSelector } from './LanguageSelector';
import { useLanguage } from '@/contexts/LanguageContext';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Badge } from '@/components/ui/badge';

interface HeaderProps {
  isAuthenticated?: boolean;
  userRole?: 'citizen' | 'officer' | 'admin';
  userName?: string;
}

export const Header = ({ isAuthenticated = false, userRole, userName }: HeaderProps) => {
  const { t } = useLanguage();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const navigate = useNavigate();

  const handleLogout = () => {
    navigate('/');
  };

  return (
    <header className="sticky top-0 z-50 w-full border-b border-border/40 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container mx-auto px-4">
        <div className="flex h-16 items-center justify-between">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-2">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary">
              <Shield className="h-6 w-6 text-primary-foreground" />
            </div>
            <div className="flex flex-col">
              <span className="text-lg font-bold text-foreground">ResolveX</span>
              <span className="hidden text-xs text-muted-foreground sm:block">E-Governance Portal</span>
            </div>
          </Link>

          {/* Desktop Navigation */}
          <nav className="hidden items-center gap-4 md:flex">
            {!isAuthenticated ? (
              <>
                <Link to="/" className="text-sm font-medium text-muted-foreground transition-colors hover:text-foreground">
                  {t('nav.home')}
                </Link>
                <LanguageSelector />
                <Button variant="ghost" asChild>
                  <Link to="/login">{t('nav.login')}</Link>
                </Button>
                <Button asChild>
                  <Link to="/register">{t('nav.register')}</Link>
                </Button>
              </>
            ) : (
              <>
                <Link to="/dashboard" className="text-sm font-medium text-muted-foreground transition-colors hover:text-foreground">
                  {t('nav.dashboard')}
                </Link>
                {userRole === 'citizen' && (
                  <>
                    <Link to="/complaints" className="text-sm font-medium text-muted-foreground transition-colors hover:text-foreground">
                      {t('nav.complaints')}
                    </Link>
                    <Link to="/track" className="text-sm font-medium text-muted-foreground transition-colors hover:text-foreground">
                      {t('nav.track')}
                    </Link>
                  </>
                )}
                <LanguageSelector />
                
                {/* Notifications */}
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="ghost" size="icon" className="relative">
                      <Bell className="h-5 w-5" />
                      <Badge className="absolute -right-1 -top-1 h-5 w-5 rounded-full p-0 text-xs">3</Badge>
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end" className="w-80">
                    <div className="p-2">
                      <h4 className="font-semibold">Notifications</h4>
                    </div>
                    <DropdownMenuSeparator />
                    <DropdownMenuItem className="flex flex-col items-start gap-1 p-3">
                      <span className="font-medium">Complaint #12345 Updated</span>
                      <span className="text-xs text-muted-foreground">Status changed to "In Progress"</span>
                      <span className="text-xs text-muted-foreground">2 hours ago</span>
                    </DropdownMenuItem>
                    <DropdownMenuItem className="flex flex-col items-start gap-1 p-3">
                      <span className="font-medium">New Response Received</span>
                      <span className="text-xs text-muted-foreground">Department officer has responded</span>
                      <span className="text-xs text-muted-foreground">5 hours ago</span>
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>

                {/* User Menu */}
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="ghost" className="gap-2">
                      <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary">
                        <User className="h-4 w-4 text-primary-foreground" />
                      </div>
                      <span className="hidden sm:inline">{userName || 'User'}</span>
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    <DropdownMenuItem>
                      <User className="mr-2 h-4 w-4" />
                      Profile
                    </DropdownMenuItem>
                    <DropdownMenuSeparator />
                    <DropdownMenuItem onClick={handleLogout}>
                      <LogOut className="mr-2 h-4 w-4" />
                      {t('nav.logout')}
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </>
            )}
          </nav>

          {/* Mobile Menu Button */}
          <Button
            variant="ghost"
            size="icon"
            className="md:hidden"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          >
            {mobileMenuOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
          </Button>
        </div>

        {/* Mobile Menu */}
        {mobileMenuOpen && (
          <div className="border-t border-border pb-4 md:hidden">
            <nav className="flex flex-col gap-2 pt-4">
              {!isAuthenticated ? (
                <>
                  <Link to="/" className="rounded-lg px-4 py-2 text-sm font-medium text-muted-foreground hover:bg-muted">
                    {t('nav.home')}
                  </Link>
                  <div className="px-4">
                    <LanguageSelector />
                  </div>
                  <Link to="/login" className="rounded-lg px-4 py-2 text-sm font-medium text-muted-foreground hover:bg-muted">
                    {t('nav.login')}
                  </Link>
                  <Link to="/register" className="rounded-lg px-4 py-2 text-sm font-medium text-primary hover:bg-muted">
                    {t('nav.register')}
                  </Link>
                </>
              ) : (
                <>
                  <Link to="/dashboard" className="rounded-lg px-4 py-2 text-sm font-medium text-muted-foreground hover:bg-muted">
                    {t('nav.dashboard')}
                  </Link>
                  <Link to="/complaints" className="rounded-lg px-4 py-2 text-sm font-medium text-muted-foreground hover:bg-muted">
                    {t('nav.complaints')}
                  </Link>
                  <Link to="/track" className="rounded-lg px-4 py-2 text-sm font-medium text-muted-foreground hover:bg-muted">
                    {t('nav.track')}
                  </Link>
                  <div className="px-4">
                    <LanguageSelector />
                  </div>
                  <button
                    onClick={handleLogout}
                    className="rounded-lg px-4 py-2 text-left text-sm font-medium text-destructive hover:bg-muted"
                  >
                    {t('nav.logout')}
                  </button>
                </>
              )}
            </nav>
          </div>
        )}
      </div>
    </header>
  );
};
