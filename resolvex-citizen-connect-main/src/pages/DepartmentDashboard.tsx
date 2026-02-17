import { useState } from 'react';
import { 
  FileText, 
  Clock, 
  CheckCircle, 
  AlertTriangle,
  Filter,
  Search,
  ChevronRight,
  User,
  ArrowUpRight
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Header } from '@/components/Header';
import { StatCard } from '@/components/StatCard';
import { useLanguage } from '@/contexts/LanguageContext';

const complaints = [
  {
    id: 'CMP-2024-12345',
    title: 'Road repair needed in residential area',
    citizen: 'John Doe',
    location: 'Karur, Tamil Nadu',
    status: 'in-progress',
    urgency: 'high',
    date: '2024-01-15',
    daysOpen: 5,
  },
  {
    id: 'CMP-2024-12346',
    title: 'Pothole causing accidents near school',
    citizen: 'Jane Smith',
    location: 'Coimbatore, Tamil Nadu',
    status: 'assigned',
    urgency: 'high',
    date: '2024-01-16',
    daysOpen: 4,
  },
  {
    id: 'CMP-2024-12347',
    title: 'Street light not functioning',
    citizen: 'Raj Kumar',
    location: 'Chennai, Tamil Nadu',
    status: 'assigned',
    urgency: 'medium',
    date: '2024-01-17',
    daysOpen: 3,
  },
  {
    id: 'CMP-2024-12348',
    title: 'Drainage overflow issue',
    citizen: 'Priya M',
    location: 'Salem, Tamil Nadu',
    status: 'received',
    urgency: 'medium',
    date: '2024-01-18',
    daysOpen: 2,
  },
  {
    id: 'CMP-2024-12349',
    title: 'Tree fallen on road',
    citizen: 'Arun K',
    location: 'Madurai, Tamil Nadu',
    status: 'received',
    urgency: 'low',
    date: '2024-01-19',
    daysOpen: 1,
  },
];

const statusOptions = [
  { value: 'all', label: 'All Status' },
  { value: 'received', label: 'Received' },
  { value: 'assigned', label: 'Assigned' },
  { value: 'in-progress', label: 'In Progress' },
  { value: 'resolved', label: 'Resolved' },
];

const urgencyOptions = [
  { value: 'all', label: 'All Priority' },
  { value: 'high', label: 'High' },
  { value: 'medium', label: 'Medium' },
  { value: 'low', label: 'Low' },
];

const statusStyles: Record<string, { label: string; className: string }> = {
  'received': { label: 'Received', className: 'bg-info/10 text-info border-info/20' },
  'assigned': { label: 'Assigned', className: 'bg-warning/10 text-warning border-warning/20' },
  'in-progress': { label: 'In Progress', className: 'bg-primary/10 text-primary border-primary/20' },
  'resolved': { label: 'Resolved', className: 'bg-success/10 text-success border-success/20' },
};

const urgencyStyles: Record<string, { label: string; className: string }> = {
  'high': { label: 'High', className: 'bg-destructive/10 text-destructive' },
  'medium': { label: 'Medium', className: 'bg-warning/10 text-warning' },
  'low': { label: 'Low', className: 'bg-muted text-muted-foreground' },
};

export const DepartmentDashboard = () => {
  const { t } = useLanguage();
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [urgencyFilter, setUrgencyFilter] = useState('all');

  const filteredComplaints = complaints.filter((complaint) => {
    const matchesSearch = complaint.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         complaint.id.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesStatus = statusFilter === 'all' || complaint.status === statusFilter;
    const matchesUrgency = urgencyFilter === 'all' || complaint.urgency === urgencyFilter;
    return matchesSearch && matchesStatus && matchesUrgency;
  });

  return (
    <div className="min-h-screen bg-background">
      <Header isAuthenticated userName="Officer Kumar" userRole="officer" />
      
      <main className="container mx-auto px-4 py-8">
        {/* Welcome Section */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-foreground">Officer Dashboard</h1>
          <p className="mt-1 text-muted-foreground">
            Public Works Department • Officer ID: <span className="font-mono font-medium text-foreground">OFF-2024-001</span>
          </p>
        </div>

        {/* Stats Grid */}
        <div className="mb-8 grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
          <StatCard
            title="Assigned to You"
            value={15}
            icon={FileText}
            variant="primary"
          />
          <StatCard
            title="Pending Action"
            value={8}
            icon={Clock}
            variant="warning"
          />
          <StatCard
            title="Resolved This Month"
            value={23}
            icon={CheckCircle}
            variant="success"
          />
          <StatCard
            title="Urgent Cases"
            value={3}
            icon={AlertTriangle}
            variant="info"
          />
        </div>

        {/* Complaints Table */}
        <Card>
          <CardHeader>
            <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <CardTitle>Assigned Complaints</CardTitle>
                <CardDescription>Manage and resolve citizen complaints</CardDescription>
              </div>
              <div className="flex flex-wrap gap-2">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                  <Input
                    placeholder="Search complaints..."
                    className="pl-9"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                  />
                </div>
                <Select value={statusFilter} onValueChange={setStatusFilter}>
                  <SelectTrigger className="w-[140px]">
                    <Filter className="mr-2 h-4 w-4" />
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {statusOptions.map((option) => (
                      <SelectItem key={option.value} value={option.value}>
                        {option.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <Select value={urgencyFilter} onValueChange={setUrgencyFilter}>
                  <SelectTrigger className="w-[140px]">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {urgencyOptions.map((option) => (
                      <SelectItem key={option.value} value={option.value}>
                        {option.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {filteredComplaints.map((complaint) => (
                <div
                  key={complaint.id}
                  className="group flex flex-col gap-4 rounded-lg border border-border p-4 transition-all hover:border-primary/50 hover:bg-muted/50 sm:flex-row sm:items-center sm:justify-between"
                >
                  <div className="flex-1 space-y-2">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="font-mono text-sm text-muted-foreground">{complaint.id}</span>
                      <Badge className={statusStyles[complaint.status].className} variant="outline">
                        {statusStyles[complaint.status].label}
                      </Badge>
                      <Badge className={urgencyStyles[complaint.urgency].className}>
                        {urgencyStyles[complaint.urgency].label} Priority
                      </Badge>
                    </div>
                    <h4 className="font-medium text-foreground">{complaint.title}</h4>
                    <div className="flex flex-wrap items-center gap-4 text-sm text-muted-foreground">
                      <span className="flex items-center gap-1">
                        <User className="h-4 w-4" />
                        {complaint.citizen}
                      </span>
                      <span>{complaint.location}</span>
                      <span>{complaint.daysOpen} days open</span>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Select defaultValue={complaint.status}>
                      <SelectTrigger className="w-[140px]">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="received">Received</SelectItem>
                        <SelectItem value="assigned">Assigned</SelectItem>
                        <SelectItem value="in-progress">In Progress</SelectItem>
                        <SelectItem value="resolved">Resolved</SelectItem>
                      </SelectContent>
                    </Select>
                    <Button variant="outline" size="icon">
                      <ArrowUpRight className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </main>
    </div>
  );
};

export default DepartmentDashboard;
