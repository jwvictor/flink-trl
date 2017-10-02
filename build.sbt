resolvers in ThisBuild ++= Seq("Apache Development Snapshot Repository" at "https://repository.apache.org/content/repositories/snapshots/",
  Resolver.mavenLocal)

name := "Flink The Regularized Leader"

version := "0.1-SNAPSHOT"

organization := "org.jasonvictor"

scalaVersion in ThisBuild := "2.11.8"

val flinkVersion = "1.3.2"

val flinkDependencies = Seq(
  "org.scalanlp" %% "breeze" % "0.12",
  "org.apache.flink" %% "flink-scala" % flinkVersion % "provided",
  "org.apache.flink" %% "flink-streaming-scala" % flinkVersion % "provided")

libraryDependencies += "joda-time" % "joda-time" % "2.9.9"

lazy val root = (project in file(".")).
  settings(
    libraryDependencies ++= flinkDependencies
  )

mainClass in assembly := Some("org.jwvictor.flinktrl.Job")

// make run command include the provided dependencies
run in Compile := Defaults.runTask(fullClasspath in Compile,
                                   mainClass in (Compile, run),
                                   runner in (Compile,run)
                                  ).evaluated

// exclude Scala library from assembly
assemblyOption in assembly := (assemblyOption in assembly).value.copy(includeScala = false)

fork in run := true
