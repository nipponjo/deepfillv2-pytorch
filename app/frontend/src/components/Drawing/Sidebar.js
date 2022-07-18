import React from "react";
import classes from "./Sidebar.module.css";

const Sidebar = (props) => {
  const saveMaskHandler = () => {
    const link = document.createElement("a");
    link.download = "mask.png";
    link.href = props.canvasRef.current.toDataURL();
    link.click();
    link.remove();
  };

  const saveImageHandler = () => {
    const link = document.createElement("a");
    link.download = "image.png";
    link.href = props.selectedImgUrl;
    link.click();
    link.remove();
  };

  return (
    <div className={classes.sidebar}>
      <label>Size:</label>
      <input
        type="number"
        onChange={(e) => {
          props.onSizeChange(+e.target.value);
        }}
        defaultValue={props.serverSize}
      />
      <button onClick={saveMaskHandler}>Save mask</button>
      <button onClick={saveImageHandler}>Save image</button>
    </div>
  );
};

export default Sidebar;
