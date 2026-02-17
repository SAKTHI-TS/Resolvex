import { 
  BarChart3, 
  TrendingUp, 
  Users, 
  Building2,
  FileText,
  Clock,
  CheckCircle,
  AlertTriangle,
  ArrowUpRight,
  ArrowDownRight,
  Activity
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Header } from '@/components/Header';
import { StatCard } from '@/components/StatCard';
import { useLanguage } from '@/contexts/LanguageContext';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line,
  Legend
} from 'recharts';

const monthlyData = [
  { month: 'Jul', received: 120, resolved: 98 },
  { month: 'Aug', received: 145, resolved: 130 },
  { month: 'Sep', received: 160, resolved: 142 },
  { month: 'Oct', received: 138, resolved: 125 },
  { month: 'Nov', received: 175, resolved: 158 },
  { month: 'Dec', received: 190, resolved: 172 },
  { month: 'Jan', received: 165, resolved: 148 },
];

const departmentData = [
  { name: 'Public Works', complaints: 450, resolved: 380, sla: 84 },
  { name: 'Water Resources', complaints: 320, resolved: 290, sla: 91 },
  { name: 'Electricity', complaints: 280, resolved: 260, sla: 93 },
  { name: 'Sanitation', complaints: 195, resolved: 175, sla: 90 },
  { name: 'Transport', complaints: 165, resolved: 140, sla: 85 },
];

const categoryDistribution = [
  { name: 'Road Issues', value: 35, color: 'hsl(var(--chart-1))' },
  { name: 'Water Supply', value: 25, color: 'hsl(var(--chart-2))' },
  { name: 'Electricity', value: 20, color: 'hsl(var(--chart-3))' },
  { name: 'Sanitation', value: 12, color: 'hsl(var(--chart-4))' },
  { name: 'Others', value: 8, color: 'hsl(var(--chart-5))' },
];

const recentActivity = [
  { action: 'New complaint registered', id: 'CMP-2024-12350', time: '5 min ago', type: 'new' },
  { action: 'Complaint resolved', id: 'CMP-2024-12321', time: '15 min ago', type: 'resolved' },
  { action: 'Escalation triggered', id: 'CMP-2024-12298', time: '30 min ago', type: 'escalated' },
  { action: 'Department assigned', id: 'CMP-2024-12349', time: '1 hour ago', type: 'assigned' },
  { action: 'Complaint resolved', id: 'CMP-2024-12315', time: '2 hours ago', type: 'resolved' },
];

export const AdminDashboard = () => {
  const { t } = useLanguage();

  return (
    <div className="min-h-screen bg-background">
      <Header isAuthenticated userName="Admin" userRole="admin" />
      
      <main className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-foreground">{t('admin.overview')}</h1>
          <p className="mt-1 text-muted-foreground">
            System-wide analytics and performance metrics
          </p>
        </div>

        {/* Key Metrics */}
        <div className="mb-8 grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
          <StatCard
            title="Total Complaints"
            value="2,847"
            icon={FileText}
            variant="primary"
            trend={{ value: 12, isPositive: true }}
          />
          <StatCard
            title="Avg Response Time"
            value="18h"
            icon={Clock}
            variant="info"
            trend={{ value: 8, isPositive: true }}
          />
          <StatCard
            title="Resolution Rate"
            value="89%"
            icon={CheckCircle}
            variant="success"
            trend={{ value: 3, isPositive: true }}
          />
          <StatCard
            title="Pending Escalations"
            value={12}
            icon={AlertTriangle}
            variant="warning"
          />
        </div>

        {/* Charts Row */}
        <div className="mb-8 grid gap-6 lg:grid-cols-2">
          {/* Trend Chart */}
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Complaint Trends</CardTitle>
                  <CardDescription>Monthly received vs resolved</CardDescription>
                </div>
                <TrendingUp className="h-5 w-5 text-muted-foreground" />
              </div>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={monthlyData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis dataKey="month" stroke="hsl(var(--muted-foreground))" fontSize={12} />
                  <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: 'hsl(var(--card))', 
                      border: '1px solid hsl(var(--border))',
                      borderRadius: '8px'
                    }}
                  />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="received" 
                    stroke="hsl(var(--primary))" 
                    strokeWidth={2}
                    name="Received"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="resolved" 
                    stroke="hsl(var(--success))" 
                    strokeWidth={2}
                    name="Resolved"
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Category Distribution */}
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Category Distribution</CardTitle>
                  <CardDescription>Complaints by category</CardDescription>
                </div>
                <BarChart3 className="h-5 w-5 text-muted-foreground" />
              </div>
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-center">
                <ResponsiveContainer width="100%" height={250}>
                  <PieChart>
                    <Pie
                      data={categoryDistribution}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={100}
                      paddingAngle={2}
                      dataKey="value"
                    >
                      {categoryDistribution.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: 'hsl(var(--card))', 
                        border: '1px solid hsl(var(--border))',
                        borderRadius: '8px'
                      }}
                      formatter={(value) => [`${value}%`, 'Share']}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-3">
                {categoryDistribution.map((cat) => (
                  <div key={cat.name} className="flex items-center gap-2">
                    <div 
                      className="h-3 w-3 rounded-full" 
                      style={{ backgroundColor: cat.color }}
                    />
                    <span className="text-sm text-muted-foreground">{cat.name}</span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Department Performance & Activity */}
        <div className="grid gap-6 lg:grid-cols-3">
          {/* Department Performance */}
          <Card className="lg:col-span-2">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Department Performance</CardTitle>
                  <CardDescription>SLA compliance and resolution metrics</CardDescription>
                </div>
                <Building2 className="h-5 w-5 text-muted-foreground" />
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                {departmentData.map((dept) => (
                  <div key={dept.name} className="space-y-2">
                    <div className="flex items-center justify-between">
                      <div>
                        <span className="font-medium text-foreground">{dept.name}</span>
                        <span className="ml-2 text-sm text-muted-foreground">
                          {dept.resolved}/{dept.complaints} resolved
                        </span>
                      </div>
                      <Badge 
                        variant={dept.sla >= 90 ? 'default' : dept.sla >= 80 ? 'secondary' : 'destructive'}
                        className={dept.sla >= 90 ? 'bg-success/10 text-success' : ''}
                      >
                        {dept.sla}% SLA
                      </Badge>
                    </div>
                    <Progress value={dept.sla} className="h-2" />
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Recent Activity */}
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>System Activity</CardTitle>
                  <CardDescription>Real-time updates</CardDescription>
                </div>
                <Activity className="h-5 w-5 text-muted-foreground" />
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {recentActivity.map((activity, index) => (
                  <div key={index} className="flex items-start gap-3">
                    <div className={`mt-1 h-2 w-2 rounded-full ${
                      activity.type === 'new' ? 'bg-info' :
                      activity.type === 'resolved' ? 'bg-success' :
                      activity.type === 'escalated' ? 'bg-destructive' :
                      'bg-primary'
                    }`} />
                    <div className="flex-1 space-y-1">
                      <p className="text-sm font-medium text-foreground">{activity.action}</p>
                      <div className="flex items-center gap-2 text-xs text-muted-foreground">
                        <span className="font-mono">{activity.id}</span>
                        <span>•</span>
                        <span>{activity.time}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Quick Stats Summary */}
        <div className="mt-6 grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
          <Card className="bg-primary/5">
            <CardContent className="flex items-center gap-4 py-4">
              <Users className="h-10 w-10 text-primary" />
              <div>
                <p className="text-2xl font-bold text-foreground">15,432</p>
                <p className="text-sm text-muted-foreground">Registered Citizens</p>
              </div>
            </CardContent>
          </Card>
          <Card className="bg-accent/10">
            <CardContent className="flex items-center gap-4 py-4">
              <Building2 className="h-10 w-10 text-accent-foreground" />
              <div>
                <p className="text-2xl font-bold text-foreground">18</p>
                <p className="text-sm text-muted-foreground">Active Departments</p>
              </div>
            </CardContent>
          </Card>
          <Card className="bg-success/5">
            <CardContent className="flex items-center gap-4 py-4">
              <ArrowUpRight className="h-10 w-10 text-success" />
              <div>
                <p className="text-2xl font-bold text-foreground">96%</p>
                <p className="text-sm text-muted-foreground">System Uptime</p>
              </div>
            </CardContent>
          </Card>
          <Card className="bg-info/5">
            <CardContent className="flex items-center gap-4 py-4">
              <BarChart3 className="h-10 w-10 text-info" />
              <div>
                <p className="text-2xl font-bold text-foreground">4.8</p>
                <p className="text-sm text-muted-foreground">Avg Satisfaction</p>
              </div>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
};

export default AdminDashboard;
