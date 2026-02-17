import { Shield, Mail, Phone, MapPin } from 'lucide-react';
import { Link } from 'react-router-dom';
export const Footer = () => {
  return <footer className="border-t border-border bg-secondary text-secondary-foreground">
      <div className="container mx-auto px-4 py-12">
        <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-4">
          {/* Brand */}
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary">
                <Shield className="h-6 w-6 text-primary-foreground" />
              </div>
              <div>
                <span className="text-lg font-bold">ResolveX</span>
              </div>
            </div>
            <p className="text-sm text-secondary-foreground/70">
              AI-Driven Multilingual Grievance Management System for transparent and efficient public service delivery.
            </p>
          </div>

          {/* Quick Links */}
          <div>
            <h4 className="mb-4 font-semibold">Quick Links</h4>
            <ul className="space-y-2 text-sm text-secondary-foreground/70">
              <li>
                <Link to="/" className="transition-colors hover:text-secondary-foreground">Home</Link>
              </li>
              <li>
                <Link to="/login" className="transition-colors hover:text-secondary-foreground">Citizen Login</Link>
              </li>
              <li>
                <Link to="/login/department" className="transition-colors hover:text-secondary-foreground">Department Login</Link>
              </li>
              <li>
                <Link to="/login/admin" className="transition-colors hover:text-secondary-foreground">Admin Login</Link>
              </li>
            </ul>
          </div>

          {/* Resources */}
          <div>
            <h4 className="mb-4 font-semibold">Resources</h4>
            <ul className="space-y-2 text-sm text-secondary-foreground/70">
              <li>
                <a href="#" className="transition-colors hover:text-secondary-foreground">User Guide</a>
              </li>
              <li>
                <a href="#" className="transition-colors hover:text-secondary-foreground">FAQ</a>
              </li>
              <li>
                <a href="#" className="transition-colors hover:text-secondary-foreground">Privacy Policy</a>
              </li>
              <li>
                <a href="#" className="transition-colors hover:text-secondary-foreground">Terms of Service</a>
              </li>
            </ul>
          </div>

          {/* Contact */}
          <div>
            
            <ul className="space-y-3 text-sm text-secondary-foreground/70">
              <li className="flex items-center gap-2">
                <MapPin className="h-4 w-4" />
                <span>M. Kumarasamy College of Engineering, Karur, TN</span>
              </li>
              <li className="flex items-center gap-2">
                <Mail className="h-4 w-4" />
                <a href="mailto:support@resolvex.gov" className="transition-colors hover:text-secondary-foreground">
                  support@resolvex.gov
                </a>
              </li>
              <li className="flex items-center gap-2">
                <Phone className="h-4 w-4" />
                <span>1800-XXX-XXXX (Toll Free)</span>
              </li>
            </ul>
          </div>
        </div>

        <div className="mt-8 border-t border-secondary-foreground/10 pt-8 text-center text-sm text-secondary-foreground/50">
          <p>© {new Date().getFullYear()} ResolveX - Government of India Initiative. All rights reserved.</p>
          <p className="mt-2">Developed with ❤️ for Smart E-Governance</p>
        </div>
      </div>
    </footer>;
};