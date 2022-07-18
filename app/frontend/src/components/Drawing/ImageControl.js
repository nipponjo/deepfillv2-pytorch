import React from "react";
import styles from "./ImageControl.module.css";
import DoubleButton from "../UI/DoubleButton";

const ImageControl = (props) => {
  return (
    <div className={styles["image-control"]}>
      <button className="button-upload">
        <label
          htmlFor={styles["image-upload"]}
          className={styles["image-upload-label"]}
        >
          Choose File
        </label>
        <input
          id={styles["image-upload"]}
          type="file"
          onChange={props.onFileSelected}
        />
      </button>
      <button className="button-clear" onClick={props.onClearClick}>
        Clear
      </button>

      <DoubleButton
        labelLeft="Undo"
        labelRight="Redo"
        onClickLeft={props.onUndoClick}
        onClickRight={props.onRedoClick}
      />

      <button
        className={styles["button-inpaint"]}
        onClick={props.onInpaintClick}
      >
        Inpaint
      </button>
    </div>
  );
};

export default ImageControl;
