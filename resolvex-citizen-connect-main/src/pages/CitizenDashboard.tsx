import { Link } from 'react-router-dom';
import { 
  FileText, 
  Search, 
  Clock, 
  CheckCircle, 
  AlertCircle,
  Plus,
  Bell,
  ChevronRight,
  TrendingUp
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Header } from '@/components/Header';
import { StatCard } from '@/components/StatCard';
import { useLanguage } from '@/contexts/LanguageContext';

const recentComplaints = [
  {
    id: 'CMP-2024-12345',
    title: 'Road repair needed in residential area',
    department: 'Public Works',
    status: 'in-progress',
    date: '2024-01-15',
  },
  {
    id: 'CMP-2024-12344',
    title: 'Street light not working',
    department: 'Electricity Board',
    status: 'resolved',
    date: '2024-01-10',
  },
  {
    id: 'CMP-2024-12343',
    title: 'Water supply issue',
    department: 'Water Resources',
    status: 'assigned',
    date: '2024-01-08',
  },
];

const notifications = [
  {
    id: 1,
    title: 'Complaint Updated',
    message: 'Your complaint CMP-2024-12345 status changed to In Progress',
    time: '2 hours ago',
    unread: true,
  },
  {
    id: 2,
    title: 'Response Received',
    message: 'Department officer has responded to your query',
    time: '5 hours ago',
    unread: true,
  },
  {
    id: 3,
    title: 'Complaint Resolved',
    message: 'Your complaint CMP-2024-12344 has been resolved',
    time: '2 days ago',
    unread: false,
  },
];

const statusStyles: Record<string, { label: string; variant: 'default' | 'secondary' | 'destructive' | 'outline' }> = {
  'submitted': { label: 'Submitted', variant: 'secondary' },
  'received': { label: 'Received', variant: 'outline' },
  'assigned': { label: 'Assigned', variant: 'outline' },
  'in-progress': { label: 'In Progress', variant: 'default' },
  'resolved': { label: 'Resolved', variant: 'secondary' },
  'closed': { label: 'Closed', variant: 'secondary' },
};

export const CitizenDashboard = () => {
  const { t } = useLanguage();

  return (
    <div className="min-h-screen bg-background">
      <Header isAuthenticated userName="John Doe" userRole="citizen" />
      
      <main className="container mx-auto px-4 py-8">
        {/* Welcome Section */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-foreground">{t('dashboard.welcome')}, John!</h1>
          <p className="mt-1 text-muted-foreground">
            User ID: <span className="font-mono font-medium text-foreground">CIT-2024-78945</span>
          </p>
        </div>

        {/* Quick Actions */}
        <div className="mb-8 grid gap-4 sm:grid-cols-2">
          <Card className="group cursor-pointer border-primary/20 bg-primary/5 transition-all hover:border-primary hover:shadow-md">
            <Link to="/citizen/file-complaint" className="block p-6">
              <div className="flex items-center gap-4">
                <div className="flex h-14 w-14 items-center justify-center rounded-xl bg-primary text-primary-foreground transition-transform group-hover:scale-110">
                  <Plus className="h-7 w-7" />
                </div>
                <div className="flex-1">
                  <h3 className="text-lg font-semibold text-foreground">{t('dashboard.newComplaint')}</h3>
                  <p className="text-sm text-muted-foreground">Submit a new grievance</p>
                </div>
                <ChevronRight className="h-5 w-5 text-muted-foreground transition-transform group-hover:translate-x-1" />
              </div>
            </Link>
          </Card>
          
          <Card className="group cursor-pointer border-accent/20 bg-accent/5 transition-all hover:border-accent hover:shadow-md">
            <Link to="/citizen/track" className="block p-6">
              <div className="flex items-center gap-4">
                <div className="flex h-14 w-14 items-center justify-center rounded-xl bg-accent text-accent-foreground transition-transform group-hover:scale-110">
                  <Search className="h-7 w-7" />
                </div>
                <div className="flex-1">
                  <h3 className="text-lg font-semibold text-foreground">{t('dashboard.trackComplaint')}</h3>
                  <p className="text-sm text-muted-foreground">View complaint status</p>
                </div>
                <ChevronRight className="h-5 w-5 text-muted-foreground transition-transform group-hover:translate-x-1" />
              </div>
            </Link>
          </Card>
        </div>

        {/* Stats Grid */}
        <div className="mb-8 grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
          <StatCard
            title={t('dashboard.total')}
            value={12}
            icon={FileText}
            variant="primary"
          />
          <StatCard
            title={t('dashboard.pending')}
            value={3}
            icon={Clock}
            variant="warning"
          />
          <StatCard
            title={t('dashboard.resolved')}
            value={8}
            icon={CheckCircle}
            variant="success"
          />
          <StatCard
            title="Urgent"
            value={1}
            icon={AlertCircle}
            variant="info"
          />
        </div>

        {/* Main Content Grid */}
        <div className="grid gap-6 lg:grid-cols-3">
          {/* Recent Complaints */}
          <Card className="lg:col-span-2">
            <CardHeader className="flex flex-row items-center justify-between">
              <div>
                <CardTitle>Recent Complaints</CardTitle>
                <CardDescription>Your latest submitted grievances</CardDescription>
              </div>
              <Button variant="ghost" size="sm" asChild>
                <Link to="/citizen/complaints" className="gap-1">
                  View All
                  <ChevronRight className="h-4 w-4" />
                </Link>
              </Button>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {recentComplaints.map((complaint) => (
                  <Link
                    key={complaint.id}
                    to={`/citizen/track/${complaint.id}`}
                    className="block rounded-lg border border-border p-4 transition-all hover:border-primary/50 hover:bg-muted/50"
                  >
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex-1 space-y-1">
                        <div className="flex items-center gap-2">
                          <span className="font-mono text-sm text-muted-foreground">{complaint.id}</span>
                          <Badge variant={statusStyles[complaint.status].variant}>
                            {statusStyles[complaint.status].label}
                          </Badge>
                        </div>
                        <h4 className="font-medium text-foreground">{complaint.title}</h4>
                        <p className="text-sm text-muted-foreground">
                          {complaint.department} • {complaint.date}
                        </p>
                      </div>
                      <ChevronRight className="h-5 w-5 text-muted-foreground" />
                    </div>
                  </Link>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Notifications */}
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <div>
                <CardTitle className="flex items-center gap-2">
                  <Bell className="h-5 w-5" />
                  Notifications
                </CardTitle>
                <CardDescription>Recent updates</CardDescription>
              </div>
              <Badge variant="secondary">3 new</Badge>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {notifications.map((notification) => (
                  <div
                    key={notification.id}
                    className={`rounded-lg p-3 transition-colors ${
                      notification.unread ? 'bg-primary/5' : 'bg-muted/50'
                    }`}
                  >
                    <div className="flex items-start gap-3">
                      {notification.unread && (
                        <div className="mt-1.5 h-2 w-2 rounded-full bg-primary" />
                      )}
                      <div className={notification.unread ? '' : 'pl-5'}>
                        <h4 className="text-sm font-medium text-foreground">{notification.title}</h4>
                        <p className="mt-1 text-sm text-muted-foreground">{notification.message}</p>
                        <p className="mt-2 text-xs text-muted-foreground">{notification.time}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Resolution Trend */}
        <Card className="mt-6">
          <CardHeader>
            <div className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5 text-primary" />
              <CardTitle>Your Resolution Trend</CardTitle>
            </div>
            <CardDescription>Complaint resolution over the past 6 months</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex h-48 items-center justify-center text-muted-foreground">
              <p>Resolution chart visualization would appear here</p>
            </div>
          </CardContent>
        </Card>
      </main>
    </div>
  );
};

export default CitizenDashboard;
