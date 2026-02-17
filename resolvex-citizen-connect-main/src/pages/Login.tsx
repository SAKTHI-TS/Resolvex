import { useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { Shield, Users, Building2, Settings, Eye, EyeOff, ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { LanguageSelector } from '@/components/LanguageSelector';
import { useLanguage } from '@/contexts/LanguageContext';

type UserRole = 'citizen' | 'department' | 'admin';

const roleConfig: Record<UserRole, {
  icon: typeof Shield;
  title: string;
  description: string;
  gradient: string;
  dashboardPath: string;
}> = {
  citizen: {
    icon: Users,
    title: 'Citizen Portal',
    description: 'Submit and track your complaints',
    gradient: 'from-primary to-primary/80',
    dashboardPath: '/citizen/dashboard',
  },
  department: {
    icon: Building2,
    title: 'Department Portal',
    description: 'Manage and resolve complaints',
    gradient: 'from-accent to-accent/80',
    dashboardPath: '/department/dashboard',
  },
  admin: {
    icon: Settings,
    title: 'Admin Portal',
    description: 'System administration and analytics',
    gradient: 'from-secondary to-secondary/80',
    dashboardPath: '/admin/dashboard',
  },
};

export const Login = () => {
  const { role = 'citizen' } = useParams<{ role?: UserRole }>();
  const currentRole = (role in roleConfig ? role : 'citizen') as UserRole;
  const config = roleConfig[currentRole];
  const Icon = config.icon;
  
  const { t } = useLanguage();
  const navigate = useNavigate();
  const [showPassword, setShowPassword] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    navigate(config.dashboardPath);
  };

  return (
    <div className="min-h-screen bg-background">
      <div className="flex min-h-screen">
        {/* Left Panel - Branding */}
        <div className={`hidden w-1/2 bg-gradient-to-br ${config.gradient} p-12 lg:flex lg:flex-col lg:justify-between`}>
          <div>
            <Link to="/" className="inline-flex items-center gap-2 text-primary-foreground/80 hover:text-primary-foreground">
              <ArrowLeft className="h-5 w-5" />
              Back to Home
            </Link>
          </div>
          
          <div className="space-y-6">
            <div className="flex items-center gap-3">
              <div className="flex h-14 w-14 items-center justify-center rounded-xl bg-primary-foreground/10 backdrop-blur-sm">
                <Shield className="h-8 w-8 text-primary-foreground" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-primary-foreground">ResolveX</h1>
                <p className="text-sm text-primary-foreground/70">E-Governance Portal</p>
              </div>
            </div>
            <div className="max-w-md">
              <h2 className="mb-4 text-4xl font-bold text-primary-foreground">
                {config.title}
              </h2>
              <p className="text-lg text-primary-foreground/80">
                {config.description}. Access our AI-powered platform for transparent and efficient grievance resolution.
              </p>
            </div>
          </div>

          <div className="text-sm text-primary-foreground/60">
            © {new Date().getFullYear()} ResolveX. Government of India Initiative.
          </div>
        </div>

        {/* Right Panel - Login Form */}
        <div className="flex w-full flex-col items-center justify-center p-8 lg:w-1/2">
          <div className="w-full max-w-md">
            {/* Mobile Header */}
            <div className="mb-8 flex items-center justify-between lg:hidden">
              <Link to="/" className="flex items-center gap-2">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary">
                  <Shield className="h-6 w-6 text-primary-foreground" />
                </div>
                <span className="font-bold">ResolveX</span>
              </Link>
              <LanguageSelector />
            </div>

            {/* Role Tabs */}
            <div className="mb-8 flex rounded-lg bg-muted p-1">
              <Link
                to="/login/citizen"
                className={`flex flex-1 items-center justify-center gap-2 rounded-md px-3 py-2 text-sm font-medium transition-colors ${
                  currentRole === 'citizen' ? 'bg-background shadow-sm' : 'text-muted-foreground hover:text-foreground'
                }`}
              >
                <Users className="h-4 w-4" />
                Citizen
              </Link>
              <Link
                to="/login/department"
                className={`flex flex-1 items-center justify-center gap-2 rounded-md px-3 py-2 text-sm font-medium transition-colors ${
                  currentRole === 'department' ? 'bg-background shadow-sm' : 'text-muted-foreground hover:text-foreground'
                }`}
              >
                <Building2 className="h-4 w-4" />
                Officer
              </Link>
              <Link
                to="/login/admin"
                className={`flex flex-1 items-center justify-center gap-2 rounded-md px-3 py-2 text-sm font-medium transition-colors ${
                  currentRole === 'admin' ? 'bg-background shadow-sm' : 'text-muted-foreground hover:text-foreground'
                }`}
              >
                <Settings className="h-4 w-4" />
                Admin
              </Link>
            </div>

            <Card className="border-0 shadow-lg">
              <CardHeader className="text-center">
                <div className={`mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-gradient-to-br ${config.gradient}`}>
                  <Icon className="h-8 w-8 text-primary-foreground" />
                </div>
                <CardTitle className="text-2xl">{t('auth.login')}</CardTitle>
                <CardDescription>{config.description}</CardDescription>
              </CardHeader>
              <CardContent>
                <form onSubmit={handleSubmit} className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="email">{t('auth.email')}</Label>
                    <Input
                      id="email"
                      type="email"
                      placeholder="you@example.com"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      required
                    />
                  </div>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <Label htmlFor="password">{t('auth.password')}</Label>
                      <a href="#" className="text-xs text-primary hover:underline">
                        {t('auth.forgot')}
                      </a>
                    </div>
                    <div className="relative">
                      <Input
                        id="password"
                        type={showPassword ? 'text' : 'password'}
                        placeholder="••••••••"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        required
                      />
                      <button
                        type="button"
                        className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                        onClick={() => setShowPassword(!showPassword)}
                      >
                        {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                      </button>
                    </div>
                  </div>
                  <Button type="submit" className="w-full">
                    {t('auth.login')}
                  </Button>
                </form>

                {currentRole === 'citizen' && (
                  <div className="mt-6 text-center text-sm text-muted-foreground">
                    Don't have an account?{' '}
                    <Link to="/register" className="text-primary hover:underline">
                      {t('auth.register')}
                    </Link>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Desktop Language Selector */}
            <div className="mt-6 hidden justify-center lg:flex">
              <LanguageSelector />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Login;
