import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Upload, FileText, X, CheckCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Header } from '@/components/Header';
import { useLanguage } from '@/contexts/LanguageContext';
import { toast } from 'sonner';

const states = [
  { value: 'tamil-nadu', label: 'Tamil Nadu' },
  { value: 'karnataka', label: 'Karnataka' },
  { value: 'kerala', label: 'Kerala' },
  { value: 'andhra-pradesh', label: 'Andhra Pradesh' },
  { value: 'maharashtra', label: 'Maharashtra' },
];

const districts: Record<string, { value: string; label: string }[]> = {
  'tamil-nadu': [
    { value: 'chennai', label: 'Chennai' },
    { value: 'coimbatore', label: 'Coimbatore' },
    { value: 'madurai', label: 'Madurai' },
    { value: 'karur', label: 'Karur' },
    { value: 'salem', label: 'Salem' },
  ],
  'karnataka': [
    { value: 'bangalore', label: 'Bangalore' },
    { value: 'mysore', label: 'Mysore' },
    { value: 'mangalore', label: 'Mangalore' },
  ],
  'kerala': [
    { value: 'thiruvananthapuram', label: 'Thiruvananthapuram' },
    { value: 'kochi', label: 'Kochi' },
    { value: 'kozhikode', label: 'Kozhikode' },
  ],
  'andhra-pradesh': [
    { value: 'hyderabad', label: 'Hyderabad' },
    { value: 'visakhapatnam', label: 'Visakhapatnam' },
    { value: 'vijayawada', label: 'Vijayawada' },
  ],
  'maharashtra': [
    { value: 'mumbai', label: 'Mumbai' },
    { value: 'pune', label: 'Pune' },
    { value: 'nagpur', label: 'Nagpur' },
  ],
};

const languages = [
  { value: 'en', label: 'English' },
  { value: 'ta', label: 'தமிழ் (Tamil)' },
  { value: 'hi', label: 'हिंदी (Hindi)' },
];

export const FileComplaint = () => {
  const { t } = useLanguage();
  const navigate = useNavigate();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [formData, setFormData] = useState({
    state: '',
    district: '',
    city: '',
    language: 'en',
    description: '',
  });
  const [attachment, setAttachment] = useState<File | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    toast.success('Complaint submitted successfully!', {
      description: 'Your complaint ID is CMP-2024-12346',
    });
    
    setIsSubmitting(false);
    navigate('/citizen/track/CMP-2024-12346');
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setAttachment(file);
    }
  };

  const removeAttachment = () => {
    setAttachment(null);
  };

  return (
    <div className="min-h-screen bg-background">
      <Header isAuthenticated userName="John Doe" userRole="citizen" />
      
      <main className="container mx-auto px-4 py-8">
        <div className="mx-auto max-w-2xl">
          <div className="mb-8">
            <h1 className="text-3xl font-bold text-foreground">{t('complaint.title')}</h1>
            <p className="mt-1 text-muted-foreground">
              Fill in the details below to submit your complaint. Our AI system will automatically route it to the appropriate department.
            </p>
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileText className="h-5 w-5" />
                Complaint Details
              </CardTitle>
              <CardDescription>
                All fields marked with * are required
              </CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-6">
                {/* User ID */}
                <div className="space-y-2">
                  <Label htmlFor="userId">{t('complaint.userId')}</Label>
                  <Input
                    id="userId"
                    value="CIT-2024-78945"
                    disabled
                    className="bg-muted font-mono"
                  />
                  <p className="text-xs text-muted-foreground">Auto-filled from your account</p>
                </div>

                {/* Location */}
                <div className="space-y-4">
                  <h3 className="font-medium text-foreground">Location Details *</h3>
                  <div className="grid gap-4 sm:grid-cols-3">
                    <div className="space-y-2">
                      <Label htmlFor="state">{t('complaint.state')}</Label>
                      <Select
                        value={formData.state}
                        onValueChange={(value) => setFormData({ ...formData, state: value, district: '' })}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Select state" />
                        </SelectTrigger>
                        <SelectContent>
                          {states.map((state) => (
                            <SelectItem key={state.value} value={state.value}>
                              {state.label}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="district">{t('complaint.district')}</Label>
                      <Select
                        value={formData.district}
                        onValueChange={(value) => setFormData({ ...formData, district: value })}
                        disabled={!formData.state}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Select district" />
                        </SelectTrigger>
                        <SelectContent>
                          {formData.state &&
                            districts[formData.state]?.map((district) => (
                              <SelectItem key={district.value} value={district.value}>
                                {district.label}
                              </SelectItem>
                            ))}
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="city">{t('complaint.city')}</Label>
                      <Input
                        id="city"
                        placeholder="Enter city/locality"
                        value={formData.city}
                        onChange={(e) => setFormData({ ...formData, city: e.target.value })}
                      />
                    </div>
                  </div>
                </div>

                {/* Language */}
                <div className="space-y-2">
                  <Label htmlFor="language">{t('complaint.language')}</Label>
                  <Select
                    value={formData.language}
                    onValueChange={(value) => setFormData({ ...formData, language: value })}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {languages.map((lang) => (
                        <SelectItem key={lang.value} value={lang.value}>
                          {lang.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-muted-foreground">
                    Choose the language for your complaint description
                  </p>
                </div>

                {/* Description */}
                <div className="space-y-2">
                  <Label htmlFor="description">{t('complaint.description')} *</Label>
                  <Textarea
                    id="description"
                    placeholder="Describe your complaint in detail..."
                    rows={6}
                    value={formData.description}
                    onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                    required
                  />
                  <p className="text-xs text-muted-foreground">
                    Please provide as much detail as possible. Our AI will analyze and categorize your complaint.
                  </p>
                </div>

                {/* Attachment */}
                <div className="space-y-2">
                  <Label>{t('complaint.attachment')}</Label>
                  {!attachment ? (
                    <div className="relative">
                      <input
                        type="file"
                        id="attachment"
                        className="absolute inset-0 cursor-pointer opacity-0"
                        accept="image/*,.pdf,.doc,.docx"
                        onChange={handleFileChange}
                      />
                      <div className="flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-border p-8 transition-colors hover:border-primary hover:bg-muted/50">
                        <Upload className="mb-2 h-10 w-10 text-muted-foreground" />
                        <p className="text-sm font-medium text-foreground">
                          Click to upload or drag and drop
                        </p>
                        <p className="text-xs text-muted-foreground">
                          PNG, JPG, PDF up to 10MB
                        </p>
                      </div>
                    </div>
                  ) : (
                    <div className="flex items-center justify-between rounded-lg border border-border bg-muted/50 p-4">
                      <div className="flex items-center gap-3">
                        <FileText className="h-8 w-8 text-primary" />
                        <div>
                          <p className="text-sm font-medium text-foreground">{attachment.name}</p>
                          <p className="text-xs text-muted-foreground">
                            {(attachment.size / 1024).toFixed(1)} KB
                          </p>
                        </div>
                      </div>
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        onClick={removeAttachment}
                      >
                        <X className="h-4 w-4" />
                      </Button>
                    </div>
                  )}
                </div>

                {/* Submit Button */}
                <Button type="submit" className="w-full" size="lg" disabled={isSubmitting}>
                  {isSubmitting ? (
                    <>
                      <div className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-primary-foreground border-t-transparent" />
                      Submitting...
                    </>
                  ) : (
                    <>
                      <CheckCircle className="mr-2 h-5 w-5" />
                      {t('complaint.submit')}
                    </>
                  )}
                </Button>
              </form>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
};

export default FileComplaint;
