import React, { createContext, useState, useContext, useEffect } from 'react';

const ProductNameContext = createContext({});

export function ProductNameProvider({ children }) {
    const [productNames, setProductNames] = useState({});
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const loadProductNames = async () => {
            try {
                console.log("Starting to load product names...");
                const response = await fetch('http://localhost:5000/api/product-names');
                
                if (!response.ok) {
                    throw new Error('Failed to fetch product names');
                }
                
                const data = await response.json();
                console.log("Received product data:", data.slice(0, 5)); // Log first 5 items
                
                // Create mapping from item_no to Product_Name
                const nameMap = {};
                data.forEach(item => {
                    if (item.item_no && item.Product_Name) {
                        nameMap[item.item_no] = item.Product_Name;
                    }
                });
                
                console.log("Created name mapping. Sample entries:", 
                    Object.entries(nameMap).slice(0, 5)); // Log first 5 mappings
                
                setProductNames(nameMap);
                setLoading(false);
            } catch (err) {
                console.error('Error loading product names:', err);
                setError(err.message);
                setLoading(false);
            }
        };

        loadProductNames();
    }, []);

    const value = {
        productNames,
        loading,
        error,
        // Add a helper function to get product name
        getProductName: (itemNo) => productNames[itemNo] || `Unknown Product (${itemNo})`
    };

    return (
        <ProductNameContext.Provider value={value}>
            {children}
        </ProductNameContext.Provider>
    );
}

export function useProductNames() {
    return useContext(ProductNameContext);
} 