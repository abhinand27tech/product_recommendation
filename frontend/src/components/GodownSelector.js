import React, { useState, useEffect, useCallback } from 'react';
import './styles.css';

const GodownSelector = ({ selectedGodown, onGodownSelect }) => {
    const [godowns, setGodowns] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [initStatus, setInitStatus] = useState({
        initialized: false,
        progress: 'Starting initialization...',
        elapsed_seconds: 0
    });

    const formatTime = (seconds) => {
        if (seconds < 60) return `${seconds} seconds`;
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = seconds % 60;
        return `${minutes} minute${minutes !== 1 ? 's' : ''} ${remainingSeconds} seconds`;
    };

    const checkInitializationStatus = useCallback(async () => {
        try {
            const response = await fetch('http://localhost:5000/api/status');
            const data = await response.json();
            
            setInitStatus({
                initialized: data.initialized,
                progress: data.progress || 'Initializing...',
                elapsed_seconds: data.elapsed_seconds || 0
            });

            if (data.initialized) {
                // Instead of calling fetchGodowns directly, we'll handle the godowns fetch here
                try {
                    setLoading(true);
                    setError(null);
                    console.log("Fetching godowns..."); // Debug log
                    
                    const godownsResponse = await fetch('http://localhost:5000/api/godowns');
                    const godownsData = await godownsResponse.json();
                    console.log("API Response:", godownsData); // Debug log

                    if (godownsResponse.status === 503) {
                        // System is still initializing
                        setInitStatus(prev => ({
                            ...prev,
                            progress: godownsData.progress || 'Still initializing...'
                        }));
                        setTimeout(checkInitializationStatus, 2000);
                        return;
                    }

                    if (godownsData.success) {
                        if (Array.isArray(godownsData.godowns) && godownsData.godowns.length > 0) {
                            setGodowns(godownsData.godowns);
                            console.log("Loaded godowns:", godownsData.godowns); // Debug log
                        } else {
                            setError('No godowns available');
                        }
                    } else {
                        setError(godownsData.error || 'Failed to fetch godowns');
                    }
                } catch (err) {
                    console.error("Error fetching godowns:", err);
                    setError('Failed to connect to the server. Please ensure the backend server is running.');
                } finally {
                    setLoading(false);
                }
            } else if (!data.error) {
                // If not initialized and no error, check again in 2 seconds
                setTimeout(checkInitializationStatus, 2000);
            }
        } catch (err) {
            console.error('Error checking initialization status:', err);
            setError('Failed to connect to the server. Please ensure the backend server is running.');
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        checkInitializationStatus();
    }, [checkInitializationStatus]);

    const handleChange = (event) => {
        const selectedValue = event.target.value;
        console.log("Selected godown:", selectedValue); // Debug log
        onGodownSelect(selectedValue);
    };

    if (!initStatus.initialized) {
        return (
            <div className="selector-container">
                <h2>System Initializing</h2>
                <div className="initialization-status">
                    <div className="loading-spinner"></div>
                    <div className="progress-info">
                        <p className="progress-message">{initStatus.progress}</p>
                        <p className="elapsed-time">Time elapsed: {formatTime(initStatus.elapsed_seconds)}</p>
                    </div>
                    <p className="init-note">
                        Initial data loading may take several minutes due to processing large datasets...
                        <br/>
                        Please keep this window open.
                    </p>
                </div>
            </div>
        );
    }

    if (loading) {
        return (
            <div className="selector-container">
                <h2>Select Supermarket</h2>
                <div className="loading-message">Loading godowns...</div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="selector-container">
                <h2>Select Supermarket</h2>
                <div className="error-message">Error: {error}</div>
            </div>
        );
    }

    return (
        <div className="selector-container">
            <h2>Select Supermarket</h2>
            <select
                value={selectedGodown || ''}
                onChange={handleChange}
                className="godown-select"
            >
                <option value="">Select a godown...</option>
                {godowns.map(godown => (
                    <option key={godown} value={godown}>
                        Godown {godown}
                    </option>
                ))}
            </select>
            {godowns.length === 0 && (
                <div className="no-data-message">No godowns available</div>
            )}
        </div>
    );
};

export default GodownSelector; 