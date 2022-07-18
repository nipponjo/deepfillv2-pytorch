import React from "react";
import classes from "./Dropzone.module.css";

const Dropzone = (props) => {
  const handleDragEnter = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };
  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };
  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  return (
    <div
      style={props.style}
      className={classes.dropzone}
      // onDrag={console.log("DRAG")}
      onDragEnter={handleDragEnter}
      // onDragStart={console.log("DRAGstart")}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={props.onDropFile}
    ></div>
  );
};

export default Dropzone;
