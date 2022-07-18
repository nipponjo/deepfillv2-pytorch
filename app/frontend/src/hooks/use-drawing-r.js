import { useState, useEffect, useRef } from "react";

const useDrawing = () => {
  const canvasRef = useRef(null);
  const contextRef = useRef(null);

  const [brushSize, setBrushSize] = useState(25);
  const [isDrawing, setIsDrawing] = useState(false);

  const currDrawingOps = useRef([]);
  const [opsBuffer, setOpsBuffer] = useState([]);
  const [opsRedoStack, setOpsRedoStack] = useState([]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas.getContext("2d");
    contextRef.current = context;

    canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
    });
  }, [canvasRef, contextRef]);

  const startDrawing = ({ nativeEvent }) => {
    contextRef.current.lineCap = "round"; // butt(def), round, square
    contextRef.current.lineJoin = "round"; // miter(def), round, bevel
    contextRef.current.strokeStyle = "#ffffff";
    contextRef.current.lineWidth = brushSize;

    const { offsetX, offsetY } = nativeEvent;
    contextRef.current.beginPath();
    contextRef.current.moveTo(offsetX, offsetY);
    setOpsBuffer((prevBuffer) => [
      ...prevBuffer,
      { type: "start", point: [offsetX, offsetY] },
    ]);
    setIsDrawing(true);
  };

  const finishDrawing = () => {
    // contextRef.current.closePath();
    const currBrushSize = contextRef.current.lineWidth;
    const ops = currDrawingOps.current.slice(0);
    setOpsBuffer((prevBuffer) => [
      ...prevBuffer,
      { type: "curve", points: ops, brushSize: currBrushSize },
    ]);
    currDrawingOps.current = [];
    setIsDrawing(false);
  };

  const draw = ({ nativeEvent }) => {
    if (!isDrawing) {
      return;
    }
    const { offsetX, offsetY } = nativeEvent;
    contextRef.current.lineTo(offsetX, offsetY);
    contextRef.current.stroke();
    currDrawingOps.current.push([offsetX, offsetY]);
  };

  const resetBuffers = () => {
    setOpsBuffer([]);
    setOpsRedoStack([]);
  };

  const clearCanvas = (clearBuffers = true) => {
    contextRef.current.clearRect(
      0,
      0,
      canvasRef.current.width,
      canvasRef.current.height
    );
    if (clearBuffers) {
      resetBuffers();
    }
  };

  const drawOps = (ops) => {
    const currBrushSize = contextRef.current.lineWidth;
    ops.forEach((op) => {
      if (op.type === "start") {
        contextRef.current.beginPath();
        contextRef.current.moveTo(...op.point);
      } else if (op.type === "curve") {
        contextRef.current.lineWidth = op.brushSize;
        op.points.forEach((point) => {
          contextRef.current.lineTo(...point);
          contextRef.current.stroke();
        });
        contextRef.current.lineWidth = currBrushSize;
      }
    });
  };

  const undoStepHandler = () => {
    if (opsBuffer.length === 0) {
      return;
    }

    clearCanvas(false);

    const newOpsBuffer = [...opsBuffer];
    const redoSteps = [];
    let lastOp;
    for (let i = 0; i < newOpsBuffer.length; i++) {
      lastOp = newOpsBuffer.pop();
      redoSteps.push(lastOp);
      if (lastOp.type === "start") {
        break;
      }
    }

    setOpsBuffer(newOpsBuffer);
    setOpsRedoStack((prevRedoStack) => [...prevRedoStack, redoSteps]);
    drawOps(newOpsBuffer);
  };

  const redoStepHandler = () => {
    if (opsRedoStack.length === 0) {
      return;
    }

    clearCanvas(false);

    const lastUndoSteps = opsRedoStack.pop();
    const newOpsBuffer = opsBuffer.concat(lastUndoSteps.reverse());

    setOpsBuffer(newOpsBuffer);
    drawOps(newOpsBuffer);
  };

  return [
    canvasRef,
    brushSize,
    setBrushSize,
    startDrawing,
    finishDrawing,
    draw,
    clearCanvas,
    undoStepHandler,
    redoStepHandler,
    resetBuffers,
  ];
};

export default useDrawing;
