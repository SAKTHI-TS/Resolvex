import { useParams, Link } from 'react-router-dom';
import { 
  ArrowLeft, 
  MapPin, 
  Building2, 
  Calendar, 
  Clock,
  RefreshCw,
  Download,
  MessageSquare
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Header } from '@/components/Header';
import { StatusTimeline } from '@/components/StatusTimeline';
import { useLanguage } from '@/contexts/LanguageContext';

const mockComplaint = {
  id: 'CMP-2024-12345',
  title: 'Road repair needed in residential area',
  description: 'There is a large pothole on the main road near Block A, Sector 15. It has been causing accidents and needs immediate attention. The pothole appeared after the recent heavy rains and has been getting worse.',
  department: 'Public Works Department',
  category: 'Road Infrastructure',
  location: {
    state: 'Tamil Nadu',
    district: 'Karur',
    city: 'Thalavapalayam',
  },
  submittedAt: '2024-01-15T10:30:00',
  updatedAt: '2024-01-18T14:45:00',
  currentStatus: 3, // 0-indexed, so this is "In Progress"
  urgency: 'High',
  officer: {
    name: 'R. Krishnan',
    designation: 'Assistant Engineer',
  },
};

const timelineSteps = [
  {
    status: 'submitted',
    label: 'Submitted',
    timestamp: 'Jan 15, 2024 10:30 AM',
    description: 'Complaint registered in the system',
  },
  {
    status: 'received',
    label: 'Received',
    timestamp: 'Jan 15, 2024 11:00 AM',
    description: 'Acknowledged by the system',
  },
  {
    status: 'assigned',
    label: 'Assigned',
    timestamp: 'Jan 16, 2024 09:15 AM',
    description: 'Assigned to Public Works Department',
  },
  {
    status: 'in-progress',
    label: 'In Progress',
    timestamp: 'Jan 18, 2024 02:45 PM',
    description: 'Work has been initiated',
  },
  {
    status: 'resolved',
    label: 'Resolved',
    timestamp: '',
    description: 'Pending resolution',
  },
  {
    status: 'closed',
    label: 'Closed',
    timestamp: '',
    description: 'Awaiting confirmation',
  },
];

export const TrackComplaint = () => {
  const { id } = useParams<{ id: string }>();
  const { t } = useLanguage();

  return (
    <div className="min-h-screen bg-background">
      <Header isAuthenticated userName="John Doe" userRole="citizen" />
      
      <main className="container mx-auto px-4 py-8">
        {/* Back Button */}
        <Link 
          to="/citizen/dashboard" 
          className="mb-6 inline-flex items-center gap-2 text-muted-foreground hover:text-foreground"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to Dashboard
        </Link>

        <div className="grid gap-6 lg:grid-cols-3">
          {/* Main Content */}
          <div className="space-y-6 lg:col-span-2">
            {/* Complaint Header */}
            <Card>
              <CardHeader>
                <div className="flex flex-wrap items-start justify-between gap-4">
                  <div>
                    <div className="mb-2 flex items-center gap-2">
                      <span className="font-mono text-sm text-muted-foreground">
                        {id || mockComplaint.id}
                      </span>
                      <Badge variant="default">In Progress</Badge>
                      <Badge variant="destructive">High Priority</Badge>
                    </div>
                    <CardTitle className="text-xl">{mockComplaint.title}</CardTitle>
                  </div>
                  <div className="flex gap-2">
                    <Button variant="outline" size="sm">
                      <RefreshCw className="mr-2 h-4 w-4" />
                      Refresh
                    </Button>
                    <Button variant="outline" size="sm">
                      <Download className="mr-2 h-4 w-4" />
                      Export
                    </Button>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="prose prose-sm max-w-none text-muted-foreground">
                  <p>{mockComplaint.description}</p>
                </div>
              </CardContent>
            </Card>

            {/* Status Timeline */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Clock className="h-5 w-5 text-primary" />
                  {t('tracking.title')}
                </CardTitle>
                <CardDescription>
                  Real-time status updates for your complaint
                </CardDescription>
              </CardHeader>
              <CardContent>
                {/* Horizontal Timeline for larger screens */}
                <div className="hidden lg:block">
                  <StatusTimeline 
                    steps={timelineSteps} 
                    currentStep={mockComplaint.currentStatus} 
                    orientation="horizontal"
                  />
                </div>
                {/* Vertical Timeline for smaller screens */}
                <div className="lg:hidden">
                  <StatusTimeline 
                    steps={timelineSteps} 
                    currentStep={mockComplaint.currentStatus} 
                    orientation="vertical"
                  />
                </div>
              </CardContent>
            </Card>

            {/* Live Update Indicator */}
            <Card className="border-primary/30 bg-primary/5">
              <CardContent className="flex items-center gap-4 py-4">
                <div className="relative">
                  <div className="h-3 w-3 rounded-full bg-primary" />
                  <div className="absolute inset-0 h-3 w-3 animate-ping rounded-full bg-primary opacity-75" />
                </div>
                <div>
                  <p className="font-medium text-foreground">Live Updates Enabled</p>
                  <p className="text-sm text-muted-foreground">
                    Last updated: {new Date(mockComplaint.updatedAt).toLocaleString()}
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* Complaint Details */}
            <Card>
              <CardHeader>
                <CardTitle>Complaint Details</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-start gap-3">
                  <Building2 className="mt-0.5 h-5 w-5 text-muted-foreground" />
                  <div>
                    <p className="text-sm text-muted-foreground">Department</p>
                    <p className="font-medium text-foreground">{mockComplaint.department}</p>
                  </div>
                </div>
                <Separator />
                <div className="flex items-start gap-3">
                  <MapPin className="mt-0.5 h-5 w-5 text-muted-foreground" />
                  <div>
                    <p className="text-sm text-muted-foreground">Location</p>
                    <p className="font-medium text-foreground">
                      {mockComplaint.location.city}, {mockComplaint.location.district}
                    </p>
                    <p className="text-sm text-muted-foreground">{mockComplaint.location.state}</p>
                  </div>
                </div>
                <Separator />
                <div className="flex items-start gap-3">
                  <Calendar className="mt-0.5 h-5 w-5 text-muted-foreground" />
                  <div>
                    <p className="text-sm text-muted-foreground">Submitted On</p>
                    <p className="font-medium text-foreground">
                      {new Date(mockComplaint.submittedAt).toLocaleDateString('en-US', {
                        year: 'numeric',
                        month: 'long',
                        day: 'numeric',
                      })}
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Assigned Officer */}
            <Card>
              <CardHeader>
                <CardTitle>Assigned Officer</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-center gap-4">
                  <div className="flex h-12 w-12 items-center justify-center rounded-full bg-primary text-primary-foreground">
                    {mockComplaint.officer.name.charAt(0)}
                  </div>
                  <div>
                    <p className="font-medium text-foreground">{mockComplaint.officer.name}</p>
                    <p className="text-sm text-muted-foreground">{mockComplaint.officer.designation}</p>
                  </div>
                </div>
                <Button variant="outline" className="mt-4 w-full">
                  <MessageSquare className="mr-2 h-4 w-4" />
                  Send Message
                </Button>
              </CardContent>
            </Card>

            {/* Need Help */}
            <Card className="bg-muted/50">
              <CardContent className="py-4">
                <h4 className="mb-2 font-medium text-foreground">Need Help?</h4>
                <p className="mb-4 text-sm text-muted-foreground">
                  If your complaint hasn't been addressed, you can escalate it to higher authorities.
                </p>
                <Button variant="outline" className="w-full">
                  Escalate Complaint
                </Button>
              </CardContent>
            </Card>
          </div>
        </div>
      </main>
    </div>
  );
};

export default TrackComplaint;
