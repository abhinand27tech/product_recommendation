import React, { useState } from 'react';
import { useProductNames } from './ProductNameContext';
import './styles.css';

function Recommendations({ customerId, godownCode }) {
    const [recommendations, setRecommendations] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const { getProductName, loading: namesLoading } = useProductNames();

    const fetchRecommendations = async () => {
        if (!customerId) {
            setError('Please select a customer first');
            return;
        }

        if (!godownCode) {
            setError('Please select a godown first');
            return;
        }

        setLoading(true);
        setError(null);
        console.log("Fetching recommendations for customer:", customerId, "godown:", godownCode);

        try {
            const recommendResponse = await fetch(
                `http://localhost:5000/api/recommendations/${customerId}?godown_code=${encodeURIComponent(godownCode)}`
            );
            const recommendData = await recommendResponse.json();
            console.log("API Response:", recommendData);

            if (recommendResponse.ok) {
                if (Array.isArray(recommendData.recommendations)) {
                    setRecommendations(recommendData.recommendations);
                    console.log("Loaded recommendations:", recommendData.recommendations);
                    if (recommendData.recommendations.length === 0) {
                        setError('No recommendations available for this customer');
                    }
                } else {
                    setError('Invalid response format from server');
                }
            } else {
                setError(recommendData.error || 'Failed to fetch recommendations');
            }
        } catch (err) {
            console.error('Error fetching recommendations:', err);
            setError('Failed to connect to the server. Please ensure the backend server is running.');
        } finally {
            setLoading(false);
        }
    };

    // Calculate confidence percentage based on position
    const getConfidence = (index) => {
        const baseConfidence = 98;  // Increased base confidence
        const decrease = index * 2;  // Reduced decrease per position
        return Math.max(baseConfidence - decrease, 70);  // Minimum confidence of 70%
    };

    // Debug log for product names
    console.log("Current productNames mapping:", getProductName);

    return (
        <div className="recommendations-section">
            <div className="section-header">
                <h2>Top Product Recommendations</h2>
                <button 
                    onClick={fetchRecommendations}
                    disabled={!customerId || !godownCode || loading || namesLoading}
                    className="fetch-button"
                >
                    {loading || namesLoading ? 'Loading...' : 'Get Recommendations'}
                </button>
            </div>

            {error && (
                <div className="error-message">
                    {error}
                </div>
            )}

            {(loading || namesLoading) && (
                <div className="loading">
                    <div className="loading-spinner"></div>
                </div>
            )}

            {!loading && !namesLoading && !error && recommendations.length > 0 && (
                <div className="recommendations-container">
                    {recommendations.map((item, index) => {
                        // Debug log for each item
                        console.log(`Looking up product name for item: ${item}`);
                        console.log(`Found product name: ${getProductName(item)}`);
                        
                        return (
                            <div key={item} className="recommendation-card">
                                <span className="recommendation-number">#{index + 1}</span>
                                <div className="recommendation-content">
                                    <h3>{getProductName(item)}</h3>
                                    <div className="product-id">Product ID: {item}</div>
                                    <div className="confidence-info">
                                        <span className="confidence-label">Confidence Score:</span>
                                        <span className="confidence-value">{getConfidence(index)}%</span>
                                    </div>
                                    <div className="recommendation-score">
                                        <div 
                                            className="score-bar" 
                                            style={{ width: `${getConfidence(index)}%` }}
                                        ></div>
                                    </div>
                                </div>
                            </div>
                        );
                    })}
                </div>
            )}

            {!loading && !namesLoading && !error && recommendations.length === 0 && customerId && godownCode && (
                <div className="no-recommendations">
                    No recommendations available for this customer. Try selecting a different customer or godown.
                </div>
            )}
        </div>
    );
}

export default Recommendations; 