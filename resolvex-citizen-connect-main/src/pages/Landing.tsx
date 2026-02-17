import { Link } from 'react-router-dom';
import { 
  Brain, 
  Globe, 
  MapPin, 
  Shield, 
  Users, 
  CheckCircle,
  ArrowRight,
  Zap,
  Lock,
  BarChart3
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Header } from '@/components/Header';
import { Footer } from '@/components/Footer';
import { useLanguage } from '@/contexts/LanguageContext';

const features = [
  {
    icon: Brain,
    titleKey: 'features.ai.title',
    descKey: 'features.ai.desc',
    color: 'primary',
  },
  {
    icon: MapPin,
    titleKey: 'features.tracking.title',
    descKey: 'features.tracking.desc',
    color: 'success',
  },
  {
    icon: Globe,
    titleKey: 'features.multilingual.title',
    descKey: 'features.multilingual.desc',
    color: 'info',
  },
];

const stats = [
  { value: '50K+', label: 'Complaints Resolved' },
  { value: '95%', label: 'Satisfaction Rate' },
  { value: '24hrs', label: 'Avg Response Time' },
  { value: '15+', label: 'Departments Connected' },
];

const capabilities = [
  { icon: Zap, title: 'Instant Routing', desc: 'AI automatically assigns complaints to the right department' },
  { icon: Lock, title: 'Secure & Private', desc: 'End-to-end encryption for all citizen data' },
  { icon: BarChart3, title: 'Analytics Dashboard', desc: 'Real-time insights for administrators' },
  { icon: Users, title: 'Multi-stakeholder', desc: 'Unified platform for citizens, officers & admins' },
];

export const Landing = () => {
  const { t } = useLanguage();

  return (
    <div className="min-h-screen bg-background">
      <Header />
      
      {/* Hero Section */}
      <section className="relative overflow-hidden gradient-hero">
        <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxnIGZpbGw9IiNmZmYiIGZpbGwtb3BhY2l0eT0iMC4wNSI+PGNpcmNsZSBjeD0iMzAiIGN5PSIzMCIgcj0iMiIvPjwvZz48L2c+PC9zdmc+')] opacity-50" />
        <div className="container relative mx-auto px-4 py-24 lg:py-32">
          <div className="mx-auto max-w-4xl text-center">
            <div className="mb-6 inline-flex items-center gap-2 rounded-full bg-primary-foreground/10 px-4 py-2 text-sm text-primary-foreground backdrop-blur-sm">
              <Shield className="h-4 w-4" />
              <span>Government of India Initiative</span>
            </div>
            <h1 className="mb-6 text-4xl font-extrabold tracking-tight text-primary-foreground sm:text-5xl lg:text-6xl animate-fade-in">
              {t('hero.title')}
            </h1>
            <p className="mb-10 text-lg text-primary-foreground/80 sm:text-xl animate-slide-up">
              {t('hero.subtitle')}
            </p>
            <div className="flex flex-col items-center justify-center gap-4 sm:flex-row animate-slide-up">
              <Button size="lg" variant="secondary" asChild className="min-w-[180px]">
                <Link to="/login">
                  {t('hero.cta.login')}
                  <ArrowRight className="ml-2 h-5 w-5" />
                </Link>
              </Button>
              <Button size="lg" variant="outline" asChild className="min-w-[180px] border-primary-foreground/30 text-primary-foreground hover:bg-primary-foreground/10">
                <Link to="/register">
                  {t('hero.cta.register')}
                </Link>
              </Button>
            </div>
          </div>
        </div>

        {/* Wave Divider */}
        <div className="absolute bottom-0 left-0 right-0">
          <svg viewBox="0 0 1440 120" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M0 120L60 110C120 100 240 80 360 70C480 60 600 60 720 65C840 70 960 80 1080 85C1200 90 1320 90 1380 90L1440 90V120H1380C1320 120 1200 120 1080 120C960 120 840 120 720 120C600 120 480 120 360 120C240 120 120 120 60 120H0Z" fill="hsl(var(--background))"/>
          </svg>
        </div>
      </section>

      {/* Stats Section */}
      <section className="bg-background py-16">
        <div className="container mx-auto px-4">
          <div className="grid grid-cols-2 gap-8 lg:grid-cols-4">
            {stats.map((stat) => (
              <div key={stat.label} className="text-center">
                <p className="text-4xl font-bold text-primary lg:text-5xl">{stat.value}</p>
                <p className="mt-2 text-sm text-muted-foreground lg:text-base">{stat.label}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="bg-muted/30 py-20">
        <div className="container mx-auto px-4">
          <div className="mx-auto mb-12 max-w-2xl text-center">
            <h2 className="mb-4 text-3xl font-bold text-foreground lg:text-4xl">
              Powered by Advanced AI Technology
            </h2>
            <p className="text-muted-foreground">
              Our platform leverages state-of-the-art NLP and machine learning to provide efficient, accurate, and transparent grievance resolution.
            </p>
          </div>

          <div className="grid gap-8 md:grid-cols-3">
            {features.map((feature) => (
              <div
                key={feature.titleKey}
                className="group rounded-2xl border border-border bg-card p-8 shadow-sm transition-all duration-300 hover:shadow-lg hover:-translate-y-1"
              >
                <div className={`mb-6 inline-flex rounded-xl bg-${feature.color}/10 p-4`}>
                  <feature.icon className={`h-8 w-8 text-${feature.color}`} />
                </div>
                <h3 className="mb-3 text-xl font-semibold text-foreground">
                  {t(feature.titleKey)}
                </h3>
                <p className="text-muted-foreground">
                  {t(feature.descKey)}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Capabilities Section */}
      <section className="bg-background py-20">
        <div className="container mx-auto px-4">
          <div className="grid gap-12 lg:grid-cols-2 lg:items-center">
            <div>
              <h2 className="mb-6 text-3xl font-bold text-foreground lg:text-4xl">
                Comprehensive Platform for Smart Governance
              </h2>
              <p className="mb-8 text-lg text-muted-foreground">
                ResolveX transforms how citizens interact with government services, ensuring every voice is heard and every complaint is addressed efficiently.
              </p>
              <div className="grid gap-6 sm:grid-cols-2">
                {capabilities.map((cap) => (
                  <div key={cap.title} className="flex gap-4">
                    <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-lg bg-primary/10">
                      <cap.icon className="h-6 w-6 text-primary" />
                    </div>
                    <div>
                      <h4 className="font-semibold text-foreground">{cap.title}</h4>
                      <p className="text-sm text-muted-foreground">{cap.desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
            <div className="relative">
              <div className="aspect-square rounded-2xl bg-gradient-to-br from-primary/20 to-accent/20 p-8">
                <div className="flex h-full flex-col items-center justify-center rounded-xl border border-border bg-card/80 backdrop-blur-sm p-8 text-center shadow-xl">
                  <Shield className="mb-4 h-16 w-16 text-primary" />
                  <h3 className="mb-2 text-2xl font-bold text-foreground">Trusted by Citizens</h3>
                  <p className="text-muted-foreground">Building transparency and accountability in public service delivery</p>
                  <div className="mt-6 flex items-center gap-2 text-success">
                    <CheckCircle className="h-5 w-5" />
                    <span className="text-sm font-medium">Government Verified Portal</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="gradient-primary py-20">
        <div className="container mx-auto px-4 text-center">
          <h2 className="mb-6 text-3xl font-bold text-primary-foreground lg:text-4xl">
            Ready to Submit Your Complaint?
          </h2>
          <p className="mx-auto mb-8 max-w-2xl text-primary-foreground/80">
            Join thousands of citizens who have found resolution through our AI-powered platform. Your voice matters.
          </p>
          <div className="flex flex-col items-center justify-center gap-4 sm:flex-row">
            <Button size="lg" variant="secondary" asChild>
              <Link to="/register">Get Started Now</Link>
            </Button>
            <Button size="lg" variant="outline" asChild className="border-primary-foreground/30 text-primary-foreground hover:bg-primary-foreground/10">
              <Link to="/login">Already Registered? Login</Link>
            </Button>
          </div>
        </div>
      </section>

      <Footer />
    </div>
  );
};

export default Landing;
