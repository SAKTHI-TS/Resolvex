import { CheckCircle2, Circle, Clock, AlertCircle } from 'lucide-react';
import { cn } from '@/lib/utils';

interface TimelineStep {
  status: string;
  label: string;
  timestamp?: string;
  description?: string;
}

interface StatusTimelineProps {
  steps: TimelineStep[];
  currentStep: number;
  orientation?: 'horizontal' | 'vertical';
}

const statusIcons = {
  completed: CheckCircle2,
  current: Clock,
  pending: Circle,
  error: AlertCircle,
};

export const StatusTimeline = ({ steps, currentStep, orientation = 'horizontal' }: StatusTimelineProps) => {
  const getStepStatus = (index: number) => {
    if (index < currentStep) return 'completed';
    if (index === currentStep) return 'current';
    return 'pending';
  };

  if (orientation === 'vertical') {
    return (
      <div className="flex flex-col space-y-0">
        {steps.map((step, index) => {
          const status = getStepStatus(index);
          const Icon = statusIcons[status];
          const isLast = index === steps.length - 1;

          return (
            <div key={step.status} className="flex gap-4">
              <div className="flex flex-col items-center">
                <div
                  className={cn(
                    'flex h-10 w-10 items-center justify-center rounded-full border-2 transition-all duration-300',
                    status === 'completed' && 'border-success bg-success text-success-foreground',
                    status === 'current' && 'border-primary bg-primary text-primary-foreground animate-pulse-soft',
                    status === 'pending' && 'border-muted-foreground/30 bg-muted text-muted-foreground'
                  )}
                >
                  <Icon className="h-5 w-5" />
                </div>
                {!isLast && (
                  <div
                    className={cn(
                      'w-0.5 flex-1 min-h-[40px] transition-all duration-300',
                      status === 'completed' ? 'bg-success' : 'bg-muted-foreground/20'
                    )}
                  />
                )}
              </div>
              <div className="flex-1 pb-8">
                <h4
                  className={cn(
                    'font-semibold',
                    status === 'completed' && 'text-success',
                    status === 'current' && 'text-primary',
                    status === 'pending' && 'text-muted-foreground'
                  )}
                >
                  {step.label}
                </h4>
                {step.timestamp && (
                  <p className="text-sm text-muted-foreground">{step.timestamp}</p>
                )}
                {step.description && (
                  <p className="mt-1 text-sm text-muted-foreground">{step.description}</p>
                )}
              </div>
            </div>
          );
        })}
      </div>
    );
  }

  return (
    <div className="flex items-center justify-between">
      {steps.map((step, index) => {
        const status = getStepStatus(index);
        const Icon = statusIcons[status];
        const isLast = index === steps.length - 1;

        return (
          <div key={step.status} className="flex flex-1 items-center">
            <div className="flex flex-col items-center">
              <div
                className={cn(
                  'flex h-12 w-12 items-center justify-center rounded-full border-2 transition-all duration-300',
                  status === 'completed' && 'border-success bg-success text-success-foreground',
                  status === 'current' && 'border-primary bg-primary text-primary-foreground animate-pulse-soft',
                  status === 'pending' && 'border-muted-foreground/30 bg-muted text-muted-foreground'
                )}
              >
                <Icon className="h-6 w-6" />
              </div>
              <div className="mt-2 text-center">
                <h4
                  className={cn(
                    'text-sm font-semibold',
                    status === 'completed' && 'text-success',
                    status === 'current' && 'text-primary',
                    status === 'pending' && 'text-muted-foreground'
                  )}
                >
                  {step.label}
                </h4>
                {step.timestamp && (
                  <p className="text-xs text-muted-foreground">{step.timestamp}</p>
                )}
              </div>
            </div>
            {!isLast && (
              <div
                className={cn(
                  'mx-2 h-1 flex-1 rounded-full transition-all duration-300',
                  status === 'completed' ? 'bg-success' : 'bg-muted-foreground/20'
                )}
              />
            )}
          </div>
        );
      })}
    </div>
  );
};
