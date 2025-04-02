import React, { useState, useEffect, useCallback } from 'react';
import './styles.css';

function CustomerSelector({ godownCode, selectedCustomer, onCustomerSelect }) {
    const [customers, setCustomers] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const fetchCustomers = useCallback(async () => {
        if (!godownCode) {
            setCustomers([]);
            return;
        }

        setLoading(true);
        setError(null);
        console.log("Fetching customers for godown:", godownCode); // Debug log

        try {
            const response = await fetch('http://localhost:5000/api/customers', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    godown_code: godownCode
                })
            });
            
            const data = await response.json();
            console.log("API Response:", data); // Debug log

            if (response.ok && data.success) {
                if (Array.isArray(data.customers) && data.customers.length > 0) {
                    setCustomers(data.customers);
                    console.log("Loaded customers:", data.customers); // Debug log
                } else {
                    setError('No customers available for this godown');
                }
            } else {
                setError(data.error || 'Failed to fetch customers');
            }
        } catch (err) {
            console.error("Error fetching customers:", err);
            setError('Failed to connect to the server. Please ensure the backend server is running.');
        } finally {
            setLoading(false);
        }
    }, [godownCode]);

    useEffect(() => {
        fetchCustomers();
    }, [fetchCustomers]);

    const handleChange = (event) => {
        const customerId = event.target.value;
        onCustomerSelect(customerId);
    };

    if (!godownCode) {
        return (
            <div className="selector-container">
                <h2>Select Customer</h2>
                <select 
                    disabled 
                    className="godown-select"
                >
                    <option>Please select a godown first...</option>
                </select>
            </div>
        );
    }

    return (
        <div className="selector-container">
            <h2>Select Customer</h2>
            {loading ? (
                <div className="loading-message">Loading customers...</div>
            ) : error ? (
                <div className="error-message">Error: {error}</div>
            ) : (
                <select
                    value={selectedCustomer || ''}
                    onChange={handleChange}
                    className="godown-select"
                >
                    <option value="">Select a customer...</option>
                    {customers.map(customer => (
                        <option key={customer} value={customer}>
                            Customer {customer}
                        </option>
                    ))}
                </select>
            )}
            {!loading && !error && customers.length === 0 && (
                <div className="no-data-message">No customers available for this godown</div>
            )}
        </div>
    );
}

export default CustomerSelector; 