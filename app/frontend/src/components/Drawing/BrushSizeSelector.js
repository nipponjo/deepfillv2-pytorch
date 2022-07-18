import { React, useState, useEffect } from "react";
import styles from "./BrushSizeSelector.module.css";

const BrushSizeSelector = (props) => {
  const [activeIdx, setActiveIdx] = useState(4);
  const [sizeChanged, setSizeChanged] = useState(null);

  useEffect(() => {
    setSizeChanged((size) => (size === null ? false : true));

    const timeout = setTimeout(() => {
      setSizeChanged(false);
    }, 1000);
    return () => {
      clearTimeout(timeout);
    };
  }, [props.brushSize]);

  const buttonClickHandler = (newBrushSize, id) => {
    props.setBrushSize(newBrushSize);
    setActiveIdx(id);
  };

  return (
    <div className={styles["width-select"]}>
      {[45, 40, 35, 30, 25, 20, 15].map((size, idx) => {
        return (
          <button
            key={size}
            style={{
              height: size,
              width: size,
              backgroundColor: idx === activeIdx ? "white" : "transparent",
            }}
            onClick={() => {
              buttonClickHandler(size, idx);
            }}
          />
        );
      })}
      <span className={sizeChanged ? styles["font-large"] : ""}>
        {props.brushSize}
      </span>
    </div>
  );
};

export default BrushSizeSelector;
