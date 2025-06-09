import { useEffect, useRef } from 'react';
import styles from '../styles/GraphView.module.css';

// Note: In a real implementation, you would use a proper graph visualization library
// like D3.js, react-force-graph, or vis-network. This is a simplified version.
const GraphView = ({ graphData }) => {
  const canvasRef = useRef(null);
  
  useEffect(() => {
    if (!graphData || !canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // Extract nodes and relationships from graphData
    const nodes = graphData.nodes || [];
    const relationships = graphData.relationships || [];
    
    // If no nodes, show a message
    if (nodes.length === 0) {
      ctx.clearRect(0, 0, width, height);
      ctx.font = '16px Arial';
      ctx.fillStyle = '#333';
      ctx.textAlign = 'center';
      ctx.fillText('No graph data available', width / 2, height / 2);
      return;
    }
    
    // Simple force-directed graph layout
    // In a real implementation, use a proper graph layout algorithm
    const nodePositions = {};
    const nodeRadius = 20;
    const nodeColors = {
      Concept: '#4CAF50', // Green
      Module: '#2196F3',  // Blue
      Function: '#FFC107', // Yellow
      Class: '#9C27B0',   // Purple
      Default: '#607D8B'  // Gray
    };
    
    // Initialize node positions randomly
    nodes.forEach((node, i) => {
      nodePositions[node.id] = {
        x: Math.random() * (width - 2 * nodeRadius) + nodeRadius,
        y: Math.random() * (height - 2 * nodeRadius) + nodeRadius,
        vx: 0,
        vy: 0
      };
    });
    
    // Simple physics simulation
    const simulate = () => {
      // Clear canvas
      ctx.clearRect(0, 0, width, height);
      
      // Draw relationships
      ctx.strokeStyle = '#999';
      ctx.lineWidth = 1;
      relationships.forEach(rel => {
        const source = nodePositions[rel.source];
        const target = nodePositions[rel.target];
        
        if (source && target) {
          ctx.beginPath();
          ctx.moveTo(source.x, source.y);
          ctx.lineTo(target.x, target.y);
          ctx.stroke();
          
          // Draw relationship type
          const midX = (source.x + target.x) / 2;
          const midY = (source.y + target.y) / 2;
          ctx.font = '10px Arial';
          ctx.fillStyle = '#666';
          ctx.textAlign = 'center';
          ctx.fillText(rel.type, midX, midY - 5);
        }
      });
      
      // Draw nodes
      nodes.forEach(node => {
        const pos = nodePositions[node.id];
        if (!pos) return;
        
        // Draw circle
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, nodeRadius, 0, 2 * Math.PI);
        ctx.fillStyle = nodeColors[node.type] || nodeColors.Default;
        ctx.fill();
        
        // Draw node name
        ctx.font = '12px Arial';
        ctx.fillStyle = '#fff';
        ctx.textAlign = 'center';
        ctx.fillText(node.name || `Node ${node.id}`, pos.x, pos.y + 4);
        
        // Draw node type
        ctx.font = '10px Arial';
        ctx.fillStyle = '#fff';
        ctx.textAlign = 'center';
        ctx.fillText(node.type, pos.x, pos.y - 8);
      });
    };
    
    // Run simulation once
    simulate();
    
    // Add interactivity
    let draggedNode = null;
    
    const getNodeAtPosition = (x, y) => {
      for (const node of nodes) {
        const pos = nodePositions[node.id];
        if (!pos) continue;
        
        const dx = pos.x - x;
        const dy = pos.y - y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        if (distance <= nodeRadius) {
          return node.id;
        }
      }
      return null;
    };
    
    const handleMouseDown = (e) => {
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      
      draggedNode = getNodeAtPosition(x, y);
    };
    
    const handleMouseMove = (e) => {
      if (!draggedNode) return;
      
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      
      nodePositions[draggedNode].x = x;
      nodePositions[draggedNode].y = y;
      
      simulate();
    };
    
    const handleMouseUp = () => {
      draggedNode = null;
    };
    
    canvas.addEventListener('mousedown', handleMouseDown);
    canvas.addEventListener('mousemove', handleMouseMove);
    canvas.addEventListener('mouseup', handleMouseUp);
    
    return () => {
      canvas.removeEventListener('mousedown', handleMouseDown);
      canvas.removeEventListener('mousemove', handleMouseMove);
      canvas.removeEventListener('mouseup', handleMouseUp);
    };
  }, [graphData]);
  
  return (
    <div className={styles.graphView}>
      <h3>Knowledge Graph</h3>
      {!graphData ? (
        <div className={styles.emptyState}>
          <p>No graph data available.</p>
          <p>Ask a question to see the knowledge graph!</p>
        </div>
      ) : (
        <div className={styles.canvasContainer}>
          <canvas
            ref={canvasRef}
            width={500}
            height={400}
            className={styles.canvas}
          />
          <div className={styles.legend}>
            <div className={styles.legendItem}>
              <span className={`${styles.legendColor} ${styles.conceptColor}`}></span>
              <span>Concept</span>
            </div>
            <div className={styles.legendItem}>
              <span className={`${styles.legendColor} ${styles.moduleColor}`}></span>
              <span>Module</span>
            </div>
            <div className={styles.legendItem}>
              <span className={`${styles.legendColor} ${styles.functionColor}`}></span>
              <span>Function</span>
            </div>
            <div className={styles.legendItem}>
              <span className={`${styles.legendColor} ${styles.classColor}`}></span>
              <span>Class</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default GraphView;