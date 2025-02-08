ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.13.16"

enablePlugins(ScalafmtPlugin)

scalafmtOnCompile := true

lazy val root = (project in file("."))
  .settings(
    name := "MLPlayground"
  )
