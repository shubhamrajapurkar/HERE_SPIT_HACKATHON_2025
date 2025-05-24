import React, { useEffect, useRef } from 'react';

interface LogEntry {
  type: 'info' | 'success' | 'warn' | 'error';
  message: string;
  timestamp?: string;
}

interface ActivityLogProps {
  logEntries: LogEntry[];
}

const ActivityLog: React.FC<ActivityLogProps> = ({ logEntries }) => {
  const logContainerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new entries are added
  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logEntries]);

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'success': return '#10B981'; // green
      case 'warn': return '#F59E0B';    // amber
      case 'error': return '#EF4444';   // red
      default: return '#287094';        // info (blue)
    }
  };

  return (
    <div
      className="fixed bottom-0 left-0 w-full z-50 bg-white shadow-lg border-t-4 flex flex-col"
      style={{ borderColor: '#287094', height: '160px' }}
    >
      {/* Header */}
      <div className="px-6 pt-3 pb-1 flex-shrink-0">
        <h3 className="text-xl font-bold" style={{ color: '#023246' }}>
          Activity Log
        </h3>
      </div>

      {/* Scrollable Content */}
      <div className="flex-1 overflow-hidden px-6 pb-3">
        <div
          ref={logContainerRef}
          className="activity-log-scroll overflow-y-auto pr-2 h-full"
          style={{
            scrollBehavior: 'smooth',
          }}
        >
          <style>
            {`
              .activity-log-scroll::-webkit-scrollbar {
                width: 6px;
                background: transparent;
              }
              .activity-log-scroll::-webkit-scrollbar-thumb {
                background: #b0b0b0;
                border-radius: 4px;
              }
              .activity-log-scroll::-webkit-scrollbar-thumb:hover {
                background: #909090;
              }
              .activity-log-scroll {
                scrollbar-width: thin;
                scrollbar-color: #b0b0b0 transparent;
              }

              @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
              }

              .animate-fadeIn {
                animation: fadeIn 0.3s ease-out;
              }
            `}
          </style>

          {/* <div className="space-y-2">
            {logEntries.length > 0 ? (
              logEntries.map((entry, index) => (
                <div
                  key={index}
                  className="p-3 rounded-lg border-l-4 text-sm animate-fadeIn"
                  style={{
                    backgroundColor: '#F6F6F6',
                    borderLeftColor: getTypeColor(entry.type),
                  }}
                >
                  <div className="flex justify-between items-start">
                    <p className="text-sm flex-1 pr-2">{entry.message}</p>
                    {entry.timestamp && (
                      <span className="text-xs opacity-50 flex-shrink-0">
                        {entry.timestamp}
                      </span>
                    )}
                  </div>
                </div>
              ))
            ) : (
              <div className="flex items-center justify-center h-full">
                <p className="text-center" style={{ color: '#D4D4CE' }}>
                  No activity yet
                </p>
              </div>
            )}
          </div> */}
          {/* Scrollable container */}
<div
  className="overflow-y-auto space-y-2"
  style={{
    maxHeight: '100px', // Adjust height to your design
    paddingRight: '4px',
    scrollbarWidth: 'thin',
  }}
>
  <style>
    {`
      .custom-scroll::-webkit-scrollbar {
        width: 6px;
      }
      .custom-scroll::-webkit-scrollbar-thumb {
        background-color: #b0b0b0;
        border-radius: 4px;
      }
    `}
  </style>

  <div className="custom-scroll">
    {logEntries.length > 0 ? (
      logEntries.map((entry, index) => (
        <div
          key={index}
          className="p-3 rounded-lg border-l-4 text-sm animate-fadeIn"
          style={{
            backgroundColor: '#F6F6F6',
            borderLeftColor: getTypeColor(entry.type),
          }}
        >
          <div className="flex justify-between items-start">
            <p className="text-sm flex-1 pr-2">{entry.message}</p>
            {entry.timestamp && (
              <span className="text-xs opacity-50 flex-shrink-0">
                {entry.timestamp}
              </span>
            )}
          </div>
        </div>
      ))
    ) : (
      <div className="flex items-center justify-center h-full">
        <p className="text-center" style={{ color: '#D4D4CE' }}>
          No activity yet
        </p>
      </div>
    )}
  </div>
</div>


        </div>
      </div>
    </div>
  );
};

export default ActivityLog;
