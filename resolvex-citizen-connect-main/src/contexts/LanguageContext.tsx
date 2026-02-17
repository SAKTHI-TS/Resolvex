import React, { createContext, useContext, useState, ReactNode } from 'react';

type Language = 'en' | 'ta' | 'hi';

interface LanguageContextType {
  language: Language;
  setLanguage: (lang: Language) => void;
  t: (key: string) => string;
}

const translations: Record<Language, Record<string, string>> = {
  en: {
    // Common
    'app.name': 'ResolveX',
    'app.tagline': 'AI-Driven Multilingual Grievance Management System',
    'app.description': 'Empowering citizens through transparent, efficient, and AI-powered complaint resolution',
    'nav.home': 'Home',
    'nav.login': 'Login',
    'nav.register': 'Register',
    'nav.dashboard': 'Dashboard',
    'nav.complaints': 'Complaints',
    'nav.track': 'Track',
    'nav.notifications': 'Notifications',
    'nav.logout': 'Logout',
    
    // Landing
    'hero.title': 'Smart Governance for a Better Tomorrow',
    'hero.subtitle': 'Submit, track, and resolve complaints in your preferred language with AI-powered efficiency',
    'hero.cta.login': 'Login to Portal',
    'hero.cta.register': 'Register Now',
    
    'features.ai.title': 'AI-Powered Routing',
    'features.ai.desc': 'Intelligent complaint categorization and automatic department assignment using advanced NLP',
    'features.tracking.title': 'Real-Time Tracking',
    'features.tracking.desc': 'Monitor your complaint status with live updates and transparent timelines',
    'features.multilingual.title': 'Multilingual Support',
    'features.multilingual.desc': 'Submit complaints in English, Tamil, or Hindi with seamless translation',
    
    // Auth
    'auth.citizen.title': 'Citizen Portal',
    'auth.citizen.desc': 'Access your citizen dashboard',
    'auth.department.title': 'Department Portal',
    'auth.department.desc': 'Officer login for complaint management',
    'auth.admin.title': 'Admin Portal',
    'auth.admin.desc': 'System administration and analytics',
    'auth.email': 'Email Address',
    'auth.password': 'Password',
    'auth.login': 'Login',
    'auth.register': 'Register',
    'auth.forgot': 'Forgot Password?',
    
    // Dashboard
    'dashboard.welcome': 'Welcome back',
    'dashboard.total': 'Total Complaints',
    'dashboard.pending': 'In Progress',
    'dashboard.resolved': 'Resolved',
    'dashboard.newComplaint': 'File New Complaint',
    'dashboard.trackComplaint': 'Track Complaints',
    
    // Complaint Form
    'complaint.title': 'File a New Complaint',
    'complaint.userId': 'User ID',
    'complaint.state': 'State',
    'complaint.district': 'District',
    'complaint.city': 'City',
    'complaint.language': 'Preferred Language',
    'complaint.description': 'Complaint Description',
    'complaint.attachment': 'Attachment (Optional)',
    'complaint.submit': 'Submit Complaint',
    
    // Tracking
    'tracking.title': 'Complaint Tracking',
    'tracking.id': 'Complaint ID',
    'tracking.department': 'Department',
    'tracking.status': 'Current Status',
    'tracking.submitted': 'Submitted',
    'tracking.received': 'Received',
    'tracking.assigned': 'Assigned',
    'tracking.inProgress': 'In Progress',
    'tracking.resolved': 'Resolved',
    'tracking.closed': 'Closed',
    
    // Admin
    'admin.overview': 'System Overview',
    'admin.analytics': 'Analytics',
    'admin.departments': 'Departments',
    'admin.performance': 'Performance',
  },
  ta: {
    // Common
    'app.name': 'ResolveX',
    'app.tagline': 'AI உந்துதல் பன்மொழி புகார் மேலாண்மை அமைப்பு',
    'app.description': 'வெளிப்படையான, திறமையான மற்றும் AI-இயக்கப்படும் புகார் தீர்வு மூலம் குடிமக்களுக்கு அதிகாரம் அளித்தல்',
    'nav.home': 'முகப்பு',
    'nav.login': 'உள்நுழை',
    'nav.register': 'பதிவு செய்',
    'nav.dashboard': 'டாஷ்போர்டு',
    'nav.complaints': 'புகார்கள்',
    'nav.track': 'கண்காணி',
    'nav.notifications': 'அறிவிப்புகள்',
    'nav.logout': 'வெளியேறு',
    
    // Landing
    'hero.title': 'சிறந்த நாளைக்கான ஸ்மார்ட் நிர்வாகம்',
    'hero.subtitle': 'AI திறனுடன் உங்கள் விருப்பமான மொழியில் புகார்களை சமர்ப்பிக்கவும், கண்காணிக்கவும், தீர்க்கவும்',
    'hero.cta.login': 'போர்டலில் உள்நுழைக',
    'hero.cta.register': 'இப்போது பதிவு செய்',
    
    'features.ai.title': 'AI-இயக்கப்படும் ரூட்டிங்',
    'features.ai.desc': 'மேம்பட்ட NLP பயன்படுத்தி புத்திசாலித்தனமான புகார் வகைப்படுத்தல்',
    'features.tracking.title': 'நிகழ்நேர கண்காணிப்பு',
    'features.tracking.desc': 'நேரடி புதுப்பிப்புகளுடன் உங்கள் புகார் நிலையை கண்காணிக்கவும்',
    'features.multilingual.title': 'பன்மொழி ஆதரவு',
    'features.multilingual.desc': 'ஆங்கிலம், தமிழ் அல்லது இந்தியில் புகார்களை சமர்ப்பிக்கவும்',
    
    // Auth
    'auth.citizen.title': 'குடிமக்கள் போர்டல்',
    'auth.citizen.desc': 'உங்கள் குடிமக்கள் டாஷ்போர்டை அணுகவும்',
    'auth.department.title': 'துறை போர்டல்',
    'auth.department.desc': 'புகார் மேலாண்மைக்கான அதிகாரி உள்நுழைவு',
    'auth.admin.title': 'நிர்வாக போர்டல்',
    'auth.admin.desc': 'கணினி நிர்வாகம் மற்றும் பகுப்பாய்வு',
    'auth.email': 'மின்னஞ்சல் முகவரி',
    'auth.password': 'கடவுச்சொல்',
    'auth.login': 'உள்நுழை',
    'auth.register': 'பதிவு செய்',
    'auth.forgot': 'கடவுச்சொல்லை மறந்துவிட்டீர்களா?',
    
    // Dashboard
    'dashboard.welcome': 'மீண்டும் வரவேற்கிறோம்',
    'dashboard.total': 'மொத்த புகார்கள்',
    'dashboard.pending': 'செயலில் உள்ளது',
    'dashboard.resolved': 'தீர்க்கப்பட்டது',
    'dashboard.newComplaint': 'புதிய புகார் பதிவு',
    'dashboard.trackComplaint': 'புகார்களை கண்காணி',
    
    // Complaint Form
    'complaint.title': 'புதிய புகாரை பதிவு செய்',
    'complaint.userId': 'பயனர் ஐடி',
    'complaint.state': 'மாநிலம்',
    'complaint.district': 'மாவட்டம்',
    'complaint.city': 'நகரம்',
    'complaint.language': 'விருப்பமான மொழி',
    'complaint.description': 'புகார் விவரம்',
    'complaint.attachment': 'இணைப்பு (விருப்பமானது)',
    'complaint.submit': 'புகாரை சமர்ப்பி',
    
    // Tracking
    'tracking.title': 'புகார் கண்காணிப்பு',
    'tracking.id': 'புகார் ஐடி',
    'tracking.department': 'துறை',
    'tracking.status': 'தற்போதைய நிலை',
    'tracking.submitted': 'சமர்ப்பிக்கப்பட்டது',
    'tracking.received': 'பெறப்பட்டது',
    'tracking.assigned': 'ஒதுக்கப்பட்டது',
    'tracking.inProgress': 'செயலில் உள்ளது',
    'tracking.resolved': 'தீர்க்கப்பட்டது',
    'tracking.closed': 'மூடப்பட்டது',
    
    // Admin
    'admin.overview': 'கணினி கண்ணோட்டம்',
    'admin.analytics': 'பகுப்பாய்வு',
    'admin.departments': 'துறைகள்',
    'admin.performance': 'செயல்திறன்',
  },
  hi: {
    // Common
    'app.name': 'ResolveX',
    'app.tagline': 'AI-संचालित बहुभाषी शिकायत प्रबंधन प्रणाली',
    'app.description': 'पारदर्शी, कुशल और AI-संचालित शिकायत समाधान के माध्यम से नागरिकों को सशक्त बनाना',
    'nav.home': 'होम',
    'nav.login': 'लॉगिन',
    'nav.register': 'रजिस्टर',
    'nav.dashboard': 'डैशबोर्ड',
    'nav.complaints': 'शिकायतें',
    'nav.track': 'ट्रैक',
    'nav.notifications': 'सूचनाएं',
    'nav.logout': 'लॉगआउट',
    
    // Landing
    'hero.title': 'बेहतर कल के लिए स्मार्ट गवर्नेंस',
    'hero.subtitle': 'AI-संचालित दक्षता के साथ अपनी पसंदीदा भाषा में शिकायतें दर्ज करें, ट्रैक करें और हल करें',
    'hero.cta.login': 'पोर्टल में लॉगिन करें',
    'hero.cta.register': 'अभी रजिस्टर करें',
    
    'features.ai.title': 'AI-संचालित रूटिंग',
    'features.ai.desc': 'उन्नत NLP का उपयोग करके बुद्धिमान शिकायत वर्गीकरण',
    'features.tracking.title': 'रियल-टाइम ट्रैकिंग',
    'features.tracking.desc': 'लाइव अपडेट के साथ अपनी शिकायत की स्थिति देखें',
    'features.multilingual.title': 'बहुभाषी सहायता',
    'features.multilingual.desc': 'अंग्रेजी, तमिल या हिंदी में शिकायतें दर्ज करें',
    
    // Auth
    'auth.citizen.title': 'नागरिक पोर्टल',
    'auth.citizen.desc': 'अपने नागरिक डैशबोर्ड तक पहुंचें',
    'auth.department.title': 'विभाग पोर्टल',
    'auth.department.desc': 'शिकायत प्रबंधन के लिए अधिकारी लॉगिन',
    'auth.admin.title': 'एडमिन पोर्टल',
    'auth.admin.desc': 'सिस्टम प्रशासन और एनालिटिक्स',
    'auth.email': 'ईमेल पता',
    'auth.password': 'पासवर्ड',
    'auth.login': 'लॉगिन',
    'auth.register': 'रजिस्टर',
    'auth.forgot': 'पासवर्ड भूल गए?',
    
    // Dashboard
    'dashboard.welcome': 'फिर से स्वागत है',
    'dashboard.total': 'कुल शिकायतें',
    'dashboard.pending': 'प्रगति में',
    'dashboard.resolved': 'हल किया गया',
    'dashboard.newComplaint': 'नई शिकायत दर्ज करें',
    'dashboard.trackComplaint': 'शिकायतें ट्रैक करें',
    
    // Complaint Form
    'complaint.title': 'नई शिकायत दर्ज करें',
    'complaint.userId': 'यूजर आईडी',
    'complaint.state': 'राज्य',
    'complaint.district': 'जिला',
    'complaint.city': 'शहर',
    'complaint.language': 'पसंदीदा भाषा',
    'complaint.description': 'शिकायत विवरण',
    'complaint.attachment': 'अटैचमेंट (वैकल्पिक)',
    'complaint.submit': 'शिकायत दर्ज करें',
    
    // Tracking
    'tracking.title': 'शिकायत ट्रैकिंग',
    'tracking.id': 'शिकायत आईडी',
    'tracking.department': 'विभाग',
    'tracking.status': 'वर्तमान स्थिति',
    'tracking.submitted': 'जमा किया गया',
    'tracking.received': 'प्राप्त हुआ',
    'tracking.assigned': 'असाइन किया गया',
    'tracking.inProgress': 'प्रगति में',
    'tracking.resolved': 'हल किया गया',
    'tracking.closed': 'बंद',
    
    // Admin
    'admin.overview': 'सिस्टम ओवरव्यू',
    'admin.analytics': 'एनालिटिक्स',
    'admin.departments': 'विभाग',
    'admin.performance': 'प्रदर्शन',
  },
};

const LanguageContext = createContext<LanguageContextType | undefined>(undefined);

export const LanguageProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [language, setLanguage] = useState<Language>('en');

  const t = (key: string): string => {
    return translations[language][key] || key;
  };

  return (
    <LanguageContext.Provider value={{ language, setLanguage, t }}>
      {children}
    </LanguageContext.Provider>
  );
};

export const useLanguage = () => {
  const context = useContext(LanguageContext);
  if (!context) {
    throw new Error('useLanguage must be used within a LanguageProvider');
  }
  return context;
};
